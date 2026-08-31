#pragma once

#include "gpu.h"

#include <stdbool.h>

// The spectral tables, baked out of process by tools/bake_spectral.py and read straight into
// one GPU buffer -- which is why there is no colour science anywhere in src/ beyond the
// blackbody normalisation below.
//
// A module level singleton rather than something the scene owns, for the same reason
// src/mesh.c is one: the shader reaches the tables through a raw device address carried in the
// push constants, so nothing about them is per-scene, and the renderer only has to ask for the
// address once.
//
// The blob holds four blocks, and the *GPU* layout is this module's choice rather than the
// file's -- the three daylight luminance constants live in the file header but are copied to
// the front of the buffer here, so the shader can reach them through the same one address
// instead of spending twelve more bytes of a nearly full push constant block:
//
//   [0]    3 floats     the daylight basis luminances Y0/Y1/Y2 (D65 itself is exactly 1)
//   [3]    9 floats     XYZ -> linear sRGB, row major
//   [12]   2 floats     the (M1, M2) that select D65 out of the daylight basis
//   [14]   1 float      the scale that gives the sun's blackbody luminance 1
//   [15]   res floats   the non-uniform z nodes for the largest-component axis
//   [14+res]   471*3    the CIE 1931 2 degree colour matching functions
//   [..]       471*3    the CIE D-series basis S0/S1/S2, pre-scaled so D65 has luminance 1
//   [..]  3*res^3*3     the Jakob-Hanika coefficients
//
// The basis carrying its own normalisation is what lets the shader's resolve be a plain sum
// against the colour matching functions with no trailing constant, and what makes a light's
// authored colour round-trip back to itself.
//
// Every offset above is a compile time constant on both sides, so the shader computes them
// from PT_SPECTRAL_RESOLUTION rather than being told. pt_spectral_init rejects any blob whose
// header disagrees, which is what keeps that safe.

// Must match the shader's own constants and the baker's defaults. The loader refuses a blob
// built with anything else and names the command that fixes it.
#define PT_SPECTRAL_RESOLUTION 64
#define PT_SPECTRAL_LAMBDA_MIN 360.0f
#define PT_SPECTRAL_LAMBDA_MAX 830.0f
#define PT_SPECTRAL_LAMBDA_COUNT 471

// The sun's colour temperature. The sky shader needs its spectrum normalised to luminance 1,
// and normalising a spectrum means integrating it against the colour matching functions --
// 471 samples the shader cannot afford per ray. So the scale is computed here at load time
// and handed over as a single number. Mirrored by PT_SUN_KELVIN in shaders/spectral.slangh.
#define PT_SPECTRAL_SUN_KELVIN 5778.0f

// Loads assets/bin/spectral.bin from `dir` and uploads it. Returns false and leaves a usable
// achromatic fallback in place when the blob is missing, stale or corrupt -- a bad asset must
// never stop the renderer from starting, and a null device address in a shader is a fault, not
// an error message. Calling twice is a no-op.
bool pt_spectral_init(gpu_device_t *device, gpu_uploader_t *uploader, const char *dir);
void pt_spectral_free(gpu_device_t *device);

// The address the push constants carry. Always valid after pt_spectral_init, fallback or not.
VkDeviceAddress pt_spectral_address(void);

// The scale that makes a blackbody of `kelvin` integrate to luminance 1, so that changing a
// light's temperature changes its colour and never its brightness -- and every intensity
// already authored keeps its meaning.
//
// Deliberately *not* the more obvious normalisation by Planck's own peak: at 2000K that peak
// sits at 1449nm, entirely outside the visible band, so peak-normalising makes a warm light
// render nearly black. Returns 1 for kelvin <= 0, which is the "no temperature authored" case.
//
// Integrated against the very same colour matching functions the shader resolves through, so
// the two cannot disagree about what luminance means.
float pt_spectral_blackbody_norm(float kelvin);
