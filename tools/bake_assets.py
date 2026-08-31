#!/usr/bin/env python3
"""Bakes every model in assets/source into a GPU-ready blob in assets/bin.

The renderer deliberately has no model parser. All the awkward work -- OBJ's index soup,
polygon triangulation, generating the normals these particular files do not carry -- happens
here, once, offline. What lands in assets/bin is a flat binary whose vertex block is already
byte-identical to pt_vertex_t, so loading it is a header check, two freads and two buffer
uploads.

Meshes are centred and uniformly scaled to fit the canonical [-1,1] object-space box, which is
the convention src/scene.h documents for every built-in shape. That is what keeps the selection
wireframe, the gizmo and "translate by scale.y to stand on the ground" working on a mesh with no
special cases, and it is what makes a 118 MB dragon and a palm-sized bunny framable by the same
instance scale.

Stdlib only, and no third party parser: the point of this script is that the repository owes
nothing to assimp or cgltf.

    python3 tools/bake_assets.py [--src DIR] [--dst DIR] [--force] [--quiet]
"""

import argparse
import math
import os
import struct
import sys
import time
from array import array

# ---------------------------------------------------------------------------
# format
# ---------------------------------------------------------------------------

MAGIC = b"PTMESH01"
VERSION = 1
HEADER_SIZE = 128  # padded, so the vertex array starts 128 byte aligned
NAME_SIZE = 64
VERTEX_SIZE = 24  # float position[3] + float normal[3], matching pt_vertex_t

# The header is little endian and so is the data behind it; the byteswap in write_mesh is what
# holds that up on a big endian host.
HEADER_FORMAT = "<8sIII3f3f3ffI%ds" % NAME_SIZE
assert struct.calcsize(HEADER_FORMAT) <= HEADER_SIZE


# ---------------------------------------------------------------------------
# obj parsing
# ---------------------------------------------------------------------------


def parse_face_corner(token, position_count, normal_count):
    """Splits one `v`, `v/vt`, `v//vn` or `v/vt/vn` corner into 0-based indices.

    OBJ indices are 1-based, and negative ones count backwards from whatever has been read so
    far -- which is why this needs the running counts rather than resolving later.
    """
    slash = token.find("/")
    if slash < 0:
        v_text, n_text = token, ""
    else:
        second = token.find("/", slash + 1)
        v_text = token[:slash]
        n_text = token[second + 1:] if second >= 0 else ""

    v = int(v_text)
    v = v - 1 if v > 0 else position_count + v

    if not n_text:
        return v, -1
    n = int(n_text)
    n = n - 1 if n > 0 else normal_count + n
    return v, n


def parse_obj(path, log):
    """Reads an OBJ into flat position/normal arrays plus a triangle corner list.

    Returns (positions, normals, corners) where `corners` holds one (position, normal) pair per
    triangle corner, normal being -1 when the file supplies none.
    """
    positions = array("f")
    normals = array("f")
    corners = []

    append_position = positions.extend
    append_normal = normals.extend
    append_corner = corners.append

    line_number = 0
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            line_number += 1

            # Cheapest possible reject first: the overwhelming majority of lines in these files
            # are `v` or `f`, and everything else -- comments, groups, materials, `vt` -- is of
            # no interest to a renderer with no textures and one material per instance.
            if not line:
                continue
            head = line[0]
            if head in " \t":
                line = line.lstrip()
                if not line:
                    continue
                head = line[0]
            if head == "v":
                fields = line.split()
                tag = fields[0]
                if tag == "v":
                    append_position((float(fields[1]), float(fields[2]), float(fields[3])))
                elif tag == "vn":
                    append_normal((float(fields[1]), float(fields[2]), float(fields[3])))
                continue
            if head != "f":
                continue

            fields = line.split()
            if fields[0] != "f":
                continue

            position_count = len(positions) // 3
            normal_count = len(normals) // 3
            try:
                face = [parse_face_corner(token, position_count, normal_count)
                        for token in fields[1:]]
            except (ValueError, IndexError):
                log("  %s:%d: skipping unparseable face" % (os.path.basename(path), line_number))
                continue

            if len(face) < 3:
                continue

            # Fan triangulation. Correct for the convex faces OBJ exporters emit, and the only
            # thing that can be done without a full ear clip -- which no file here needs.
            first = face[0]
            for i in range(1, len(face) - 1):
                append_corner(first)
                append_corner(face[i])
                append_corner(face[i + 1])

    return positions, normals, corners


# ---------------------------------------------------------------------------
# vertex building
# ---------------------------------------------------------------------------


def build_with_normals(positions, normals, corners):
    """Uses the file's own normals, keying unique vertices on the (position, normal) pair.

    A shared position with two different normals is a hard edge and has to stay two vertices;
    merging them would round the crease off.
    """
    unique = {}
    vertices = array("f")
    indices = array("I")

    append_vertex = vertices.extend
    append_index = indices.append

    for corner in corners:
        index = unique.get(corner)
        if index is None:
            index = len(unique)
            unique[corner] = index
            p = corner[0] * 3
            n = corner[1] * 3
            append_vertex((positions[p], positions[p + 1], positions[p + 2],
                           normals[n], normals[n + 1], normals[n + 2]))
        append_index(index)

    return vertices, indices


def build_smooth_normals(positions, corners, log):
    """Generates area-weighted smoothed vertex normals.

    None of the three models in assets/source carries a single `vn` line, but closesthit_mesh
    interpolates a normal per hit, so one has to be invented. Accumulating the *un-normalised*
    face cross product is what makes this area weighted for free: its length is twice the
    triangle's area, so large triangles pull on a shared vertex proportionally more than the
    slivers around them do.

    This is the slow part of the bake -- 2.4 million triangles of pure Python -- hence the flat
    arrays and the hoisted locals rather than anything prettier.
    """
    vertex_count = len(positions) // 3
    vertices = array("f", bytes(vertex_count * VERTEX_SIZE))
    indices = array("I", bytes(len(corners) * 4))

    # Positions first, interleaved into their final slots; the normals accumulate in place.
    for i in range(vertex_count):
        p = i * 3
        v = i * 6
        vertices[v] = positions[p]
        vertices[v + 1] = positions[p + 1]
        vertices[v + 2] = positions[p + 2]

    degenerate = 0
    for triangle in range(len(corners) // 3):
        base = triangle * 3
        ia = corners[base][0]
        ib = corners[base + 1][0]
        ic = corners[base + 2][0]

        indices[base] = ia
        indices[base + 1] = ib
        indices[base + 2] = ic

        a = ia * 3
        b = ib * 3
        c = ic * 3
        ax = positions[a]
        ay = positions[a + 1]
        az = positions[a + 2]
        ux = positions[b] - ax
        uy = positions[b + 1] - ay
        uz = positions[b + 2] - az
        vx = positions[c] - ax
        vy = positions[c + 1] - ay
        vz = positions[c + 2] - az

        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        if nx == 0.0 and ny == 0.0 and nz == 0.0:
            degenerate += 1
            continue

        na = ia * 6 + 3
        vertices[na] += nx
        vertices[na + 1] += ny
        vertices[na + 2] += nz
        nb = ib * 6 + 3
        vertices[nb] += nx
        vertices[nb + 1] += ny
        vertices[nb + 2] += nz
        nc = ic * 6 + 3
        vertices[nc] += nx
        vertices[nc + 1] += ny
        vertices[nc + 2] += nz

    if degenerate:
        log("  %d degenerate triangle(s) contributed no normal" % degenerate)

    orphans = 0
    for i in range(vertex_count):
        n = i * 6 + 3
        nx = vertices[n]
        ny = vertices[n + 1]
        nz = vertices[n + 2]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-20:
            # A vertex no non-degenerate triangle touched. Any unit vector will do; it is never
            # interpolated because no primitive references it.
            vertices[n] = 0.0
            vertices[n + 1] = 1.0
            vertices[n + 2] = 0.0
            orphans += 1
            continue
        inverse = 1.0 / length
        vertices[n] = nx * inverse
        vertices[n + 1] = ny * inverse
        vertices[n + 2] = nz * inverse

    if orphans:
        log("  %d vertex/vertices had no incident triangle" % orphans)

    return vertices, indices


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def bounds_of(vertices):
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for i in range(0, len(vertices), 6):
        for axis in range(3):
            value = vertices[i + axis]
            if value < lo[axis]:
                lo[axis] = value
            if value > hi[axis]:
                hi[axis] = value
    return lo, hi


def normalize(vertices):
    """Centres the mesh and uniformly scales it to fit [-1,1].

    Uniform, so proportions survive: the longest axis spans exactly [-1,1] and the other two sit
    inside it. Returns the centre and the scale so the transform stays recorded in the header
    rather than being silently lost.
    """
    lo, hi = bounds_of(vertices)
    center = [(lo[axis] + hi[axis]) * 0.5 for axis in range(3)]
    extent = max(hi[axis] - lo[axis] for axis in range(3))
    scale = 2.0 / extent if extent > 0.0 else 1.0

    for i in range(0, len(vertices), 6):
        vertices[i] = (vertices[i] - center[0]) * scale
        vertices[i + 1] = (vertices[i + 1] - center[1]) * scale
        vertices[i + 2] = (vertices[i + 2] - center[2]) * scale

    return center, scale


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_mesh(path, name, vertices, indices, aabb_min, aabb_max, center, scale):
    header = struct.pack(HEADER_FORMAT, MAGIC, VERSION,
                         len(vertices) // 6, len(indices),
                         aabb_min[0], aabb_min[1], aabb_min[2],
                         aabb_max[0], aabb_max[1], aabb_max[2],
                         center[0], center[1], center[2],
                         scale, 0,
                         name.encode("utf-8")[:NAME_SIZE - 1])

    # Written to a temporary and renamed, so an interrupted bake can never leave a truncated
    # file behind that the loader would then have to be defensive about at startup.
    temporary = path + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(header)
        handle.write(b"\0" * (HEADER_SIZE - len(header)))
        if sys.byteorder == "big":
            vertices = array("f", vertices)
            indices = array("I", indices)
            vertices.byteswap()
            indices.byteswap()
        vertices.tofile(handle)
        indices.tofile(handle)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def bake(source, destination, log):
    started = time.monotonic()
    name = os.path.splitext(os.path.basename(source))[0].lower()

    log("baking %s" % os.path.basename(source))
    positions, normals, corners = parse_obj(source, log)

    if not corners:
        log("  no triangles, skipped")
        return False

    if normals and any(corner[1] >= 0 for corner in corners):
        vertices, indices = build_with_normals(positions, normals, corners)
        source_of_normals = "from file"
    else:
        vertices, indices = build_smooth_normals(positions, corners, log)
        source_of_normals = "generated"

    lo, hi = bounds_of(vertices)
    center, scale = normalize(vertices)
    baked_min, baked_max = bounds_of(vertices)

    write_mesh(destination, name, vertices, indices, baked_min, baked_max, center, scale)

    log("  %d vertices, %d triangles, normals %s"
        % (len(vertices) // 6, len(indices) // 3, source_of_normals))
    log("  source bounds  [%.4g %.4g %.4g] .. [%.4g %.4g %.4g]"
        % (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]))
    log("  baked aabb     [%.4f %.4f %.4f] .. [%.4f %.4f %.4f]  (scale %.6g)"
        % (baked_min[0], baked_min[1], baked_min[2],
           baked_max[0], baked_max[1], baked_max[2], scale))
    log("  -> %s (%.1f MB, %.1fs)"
        % (os.path.basename(destination),
           os.path.getsize(destination) / (1024.0 * 1024.0),
           time.monotonic() - started))
    return True


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=os.path.join(root, "assets", "source"))
    parser.add_argument("--dst", default=os.path.join(root, "assets", "bin"))
    parser.add_argument("--force", action="store_true",
                        help="re-bake even when the output is already up to date")
    parser.add_argument("--verbose", action="store_true",
                        help="also report the models that were already up to date")
    args = parser.parse_args()

    def log(message):
        print(message)
        sys.stdout.flush()

    # A warm build should say nothing at all, so the models that needed no work are only
    # mentioned when asked for. Anything actually baked always reports.
    def detail(message):
        if args.verbose:
            log(message)

    if not os.path.isdir(args.src):
        print("bake_assets: no source directory '%s'" % args.src, file=sys.stderr)
        return 0  # not an error: a checkout without models still has to build

    os.makedirs(args.dst, exist_ok=True)

    sources = sorted(entry for entry in os.listdir(args.src) if entry.lower().endswith(".obj"))
    if not sources:
        detail("bake_assets: nothing to bake in %s" % args.src)
        return 0

    baked = 0
    failed = 0
    for entry in sources:
        source = os.path.join(args.src, entry)
        destination = os.path.join(args.dst,
                                   os.path.splitext(entry)[0].lower() + ".ptm")

        if not args.force and os.path.exists(destination) and \
                os.path.getmtime(destination) >= os.path.getmtime(source):
            detail("up to date %s" % os.path.basename(destination))
            continue

        try:
            if bake(source, destination, log):
                baked += 1
        except Exception as error:  # one bad model must not stop the others
            print("bake_assets: %s failed: %s" % (entry, error), file=sys.stderr)
            failed += 1

    if baked:
        log("bake_assets: baked %d model(s)" % baked)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
