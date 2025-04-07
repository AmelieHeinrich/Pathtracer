//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 22:05:34
//

#include "Skybox.hpp"

SkyboxCooker::Data SkyboxCooker::sData;

void SkyboxCooker::Init()
{
    ShaderFile file = ShaderCompiler::Load("Shaders/SkyboxGeneration.hlsl");
    auto sig = std::make_shared<RootSignature>(std::vector<RootType>{ RootType::PushConstant }, sizeof(glm::vec4));
    sData.ConvertCubemapPipeline = std::make_shared<ComputePipeline>(file.Modules["CSMain"], sig);

    sData.ConvertSampler = std::make_shared<Sampler>(SamplerAddress::Wrap, SamplerFilter::Linear);
}

void SkyboxCooker::Exit()
{
    sData.ConvertCubemapPipeline.reset();
    sData.ConvertSampler.reset();
}

std::shared_ptr<Skybox> SkyboxCooker::LoadSkybox(const std::string& path)
{
    // Load HDR skybox
    ImageData data;
    data.LoadHDR(path);

    TextureDesc desc;
    desc.Width = data.Width;
    desc.Height = data.Height;
    desc.Depth = 1;
    desc.Levels = 1;
    desc.Usage = TextureUsage::ShaderResource;
    desc.Format = TextureFormat::RGBA16Unorm;
    desc.Name = path;
    std::shared_ptr<Texture> stagingTexture = std::make_shared<Texture>(desc);

    Uploader::EnqueueTextureUpload(data.Pixels, stagingTexture);
    Uploader::Flush();

    // Create cubnemap
    std::shared_ptr<Skybox> skybox = std::make_shared<Skybox>();
    
    desc.Width = 512;
    desc.Height = 512;
    desc.Depth = 6;
    desc.Levels = 1;
    desc.Usage = TextureUsage::Storage | TextureUsage::ShaderResource;
    desc.Name = path;
    desc.Format = TextureFormat::RGBA16Unorm;
    skybox->SkyboxTexture = std::make_shared<Texture>(desc);
    skybox->SkyboxCubeView = std::make_shared<View>(skybox->SkyboxTexture, ViewType::ShaderResource, ViewDimension::TextureCube);

    // Create views
    std::shared_ptr<View> hdrSRV = std::make_shared<View>(stagingTexture, ViewType::ShaderResource);
    std::shared_ptr<View> envUAV = std::make_shared<View>(skybox->SkyboxTexture, ViewType::Storage, ViewDimension::TextureCube);

    // Convert from equilateral to cubemap
    struct Data {
        int hdrIn;
        int envOut;
        int sampler;
        int pad;
    } constants = {
        hdrSRV->GetDescriptor().Index,
        envUAV->GetDescriptor().Index,
        sData.ConvertSampler->BindlesssSampler(),
        0
    };

    std::shared_ptr<CommandBuffer> cmdBuffer = std::make_shared<CommandBuffer>(RHI::GetGraphicsQueue(), true);
    cmdBuffer->Begin();
    cmdBuffer->SetComputePipeline(sData.ConvertCubemapPipeline);
    cmdBuffer->ComputePushConstants(&constants, sizeof(constants), 0);
    cmdBuffer->Dispatch(512 / 32, 512 / 32, 6);
    cmdBuffer->End();

    RHI::Submit({ cmdBuffer });
    RHI::Wait();

    return skybox;
}
