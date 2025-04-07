--
-- > Notice: Amélie Heinrich @ 2025
-- > Create Time: 2025-04-06 10:38:15
--

add_rules("mode.debug", "mode.release", "mode.releasedbg")

includes("Oslo")

target("Pathtracer")
    set_rundir(".")
    set_kind("binary")

    add_files("Source/**.cpp")
    add_includedirs("Oslo", "Source")
    add_deps("Oslo")

    before_link(function (target)
        os.cp("Oslo/Binaries/*", "$(buildir)/$(plat)/$(arch)/$(mode)/")
    end)
