//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-10 22:57:57
//

#pragma once

#include "Renderer/Pass.hpp"
#include "Renderer/Skybox.hpp"

class TonemapPass : public Pass
{
public:
    TonemapPass();

    void Render(Frame& frame, Scene& scene) override;
    void UI() override;

private:
    std::shared_ptr<ComputePipeline> mPipeline;
    float mGamma = 2.2f;
};
