//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:47:20
//

#pragma once

#include "Renderer/Pass.hpp"
#include "Renderer/Skybox.hpp"

class MainPass : public Pass
{
public:
    MainPass();

    void Render(Frame& frame, Scene& scene) override;
    void UI() override;

private:
    std::shared_ptr<RaytracingPipeline> mPipeline;
    std::shared_ptr<Skybox> mSkybox;

    int mFrameIndex = 0;
    int mSamplesPerPixel = 1;
    int mBouncesPerRay = 5;
};
