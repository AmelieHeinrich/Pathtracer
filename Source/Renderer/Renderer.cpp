//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:41:28
//

#include "Renderer.hpp"
#include "RendererTools.hpp"

#include "Cache/TextureCache.hpp"

#include "Passes/MainPass.hpp"
#include "Passes/ResolvePass.hpp"
#include "Passes/TonemapPass.hpp"

#include "Skybox.hpp"

Renderer::Renderer()
{
    RendererTools::Init();
    SkyboxCooker::Init();

    mPasses = {
        std::make_shared<MainPass>(),
        std::make_shared<ResolvePass>(),
        std::make_shared<TonemapPass>()
    };
}

Renderer::~Renderer()
{
    mPasses.clear();
    SkyboxCooker::Exit();
    RendererTools::Free();
    TextureCache::Clear();
}

void Renderer::Render(Frame& frame, Scene& scene)
{
    for (auto& pass : mPasses) {
        pass->Render(frame, scene);
    }
}

void Renderer::UI()
{
    for (auto& pass : mPasses) {
        pass->UI();
    }
}
