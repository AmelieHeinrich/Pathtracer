//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:41:28
//

#include "Renderer.hpp"
#include "RendererTools.hpp"

#include "Passes/MainPass.hpp"

Renderer::Renderer()
{
    mPasses = {
        std::make_shared<MainPass>()
    };
    RendererTools::Init();
}

Renderer::~Renderer()
{
    mPasses.clear();
    RendererTools::Free();
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
