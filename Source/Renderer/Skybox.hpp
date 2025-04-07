//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 22:01:09
//

#pragma once

#include <Oslo/Oslo.hpp>

struct Skybox
{
    std::shared_ptr<Texture> SkyboxTexture;
    std::shared_ptr<View> SkyboxCubeView;
};

class SkyboxCooker
{
public:
    static void Init();
    static void Exit();
    
    static std::shared_ptr<Skybox> LoadSkybox(const std::string& path);
private:
    static struct Data {
        std::shared_ptr<Sampler> ConvertSampler;
        std::shared_ptr<ComputePipeline> ConvertCubemapPipeline;
    } sData;
};
