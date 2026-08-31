-- xmake.lua

add_rules("mode.debug", "mode.release", "mode.releasedbg")
add_requires("slang", {configs = {slangc = true}})
add_requires("vulkan-headers", "volk", "cgltf", "stb", "glfw", "nuklear")

set_rundir(".")

target("Pathtracer")
    set_kind("binary")
    set_languages("c11")
    set_warnings("all")

    if is_mode("debug", "releasedbg") then
        add_defines("PT_ENABLE_VALIDATION")
    end

    -- nuklear.h's single-header implementation does not compile clean under -Wall, so it
    -- gets a translation unit of its own with warnings off rather than polluting the build.
    add_files("src/*.c|nuklear_impl.c")
    add_files("src/nuklear_impl.c", {cflags = "-w"})
    -- slang is used only as a build-time and hot-reload *tool* (slangc is spawned as a
    -- process), so it is deliberately linked with no libraries: linking libslang.so would
    -- add a runtime dependency on its plugin .so files for no benefit.
    add_packages("slang", {links = {}})
    add_packages("vulkan-headers", "volk", "cgltf", "stb", "glfw", "nuklear")

    -- Bake the shader toolchain paths in, so the build rule below and the runtime hot
    -- reload in renderer.c always agree on where slangc and the shaders live.
    on_load(function (target)
        local slangc = "slangc"
        local slang = target:pkg("slang")
        if slang and slang:installdir() then
            slangc = path.join(slang:installdir(), "bin", "slangc")
        end
        target:data_set("slangc", slangc)

        local spvdir = path.absolute(path.join(target:targetdir(), "shaders"))
        target:add("defines", 'PT_SLANGC="' .. slangc .. '"')
        target:add("defines", 'PT_SHADER_DIR="' .. spvdir .. '"')
        target:add("defines", 'PT_SHADER_SRC_DIR="' .. path.absolute("shaders") .. '"')
        -- Scenes are authored in place and saved back over the source tree, so this points at
        -- the repository rather than at anything under the build directory.
        target:add("defines", 'PT_SCENE_DIR="' .. path.absolute("scenes") .. '"')
    end)

    -- One .slang module compiles to one .spv holding every entry point.
    --   -fvk-use-entrypoint-name : without it Slang renames every entry point to "main",
    --                              which makes a multi-stage ray tracing module impossible
    --   -fvk-use-scalar-layout   : makes Slang struct layouts match the C structs exactly
    before_build(function (target)
        import("core.base.option")
        local slangc = target:data("slangc")
        local outdir = path.join(target:targetdir(), "shaders")
        os.mkdir(outdir)

        -- .slangh files are #include'd rather than compiled on their own, so they never
        -- produce a .spv, but editing one still has to force a rebuild.
        local newest_header = 0
        for _, header in ipairs(os.files("shaders/*.slangh")) do
            newest_header = math.max(newest_header, os.mtime(header))
        end

        for _, source in ipairs(os.files("shaders/*.slang")) do
            local output = path.join(outdir, path.basename(source) .. ".spv")
            local newest = math.max(os.mtime(source), newest_header)
            if not os.isfile(output) or newest > os.mtime(output) then
                cprint("${color.build.object}compiling.shader %s", source)
                -- -O0 because this slang build has no slang-glslang, so its internal
                -- spirv-opt cannot load. The system spirv-opt below does the job instead.
                os.vrunv(slangc, {source,
                                  "-target", "spirv",
                                  "-profile", "spirv_1_5",
                                  "-emit-spirv-directly",
                                  "-fvk-use-entrypoint-name",
                                  "-fvk-use-scalar-layout",
                                  "-O0",
                                  "-o", output})
                -- --scalar-block-layout or its built-in validator rejects the tightly
                -- packed float3 members that -fvk-use-scalar-layout produces.
                if os.isexec("/usr/bin/spirv-opt") then
                    os.vrunv("/usr/bin/spirv-opt",
                             {"--scalar-block-layout", "-O", output, "-o", output})
                end
            end
        end
    end)
