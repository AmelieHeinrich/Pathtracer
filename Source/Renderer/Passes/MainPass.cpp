//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 17:12:42
//

#include "MainPass.hpp"
#include "Renderer/RendererTools.hpp"

#include <imgui.h>

struct RayPayload
{
    glm::vec3 Throughput;
    glm::vec3 AccumulatedColor;

    int Bounce;
    bool Alive;
    glm::vec3 NewDirection;
    glm::vec3 NewOrigin;

    glm::uvec2 rng;
};

MainPass::MainPass()
{
    ShaderFile file = ShaderCompiler::Load("Shaders/Raytrace.hlsl");

    RaytracingPipelineSpecs specs = {};
    specs.AttribSize = sizeof(glm::vec2);
    specs.MaxRecursion = 3;
    specs.PayloadSize = sizeof(RayPayload);
    specs.Library = file.Modules["Shader"];
    specs.Signature = std::make_shared<RootSignature>(std::vector<RootType>{RootType::PushConstant }, sizeof(glm::ivec4) * 3);

    mPipeline = std::make_shared<RaytracingPipeline>(specs);

    TextureDesc desc = {};
    desc.Width = 1280;
    desc.Height = 720;
    desc.Depth = 1;
    desc.Levels = 1;
    desc.Usage = TextureUsage::Storage;
    desc.Format = TextureFormat::RGBA8;
    desc.Name = "RT Output";

    auto tex = RendererTools::CreateSharedTexture("RTOutput", desc);
    tex->AddView(ViewType::Storage, ViewDimension::Texture);

    RendererTools::CreateSharedRingBuffer("CameraBuffer", 256, 0);
    RendererTools::CreateSharedSampler("TextureSampler", SamplerFilter::Linear, SamplerAddress::Wrap);

    mSkybox = SkyboxCooker::LoadSkybox("Assets/Skybox/Garden.hdr");
}

void MainPass::Render(Frame& frame, Scene& scene)
{
    auto out = RendererTools::Get("RTOutput");
    auto cam = RendererTools::Get("CameraBuffer");
    auto sampler = RendererTools::Get("TextureSampler");

    struct CameraData {
        glm::mat4 invView;
        glm::mat4 invProj;
        glm::vec3 cameraPos;
        int pad;
    } camData = {
        glm::inverse(scene.CamInfo.View),
        glm::inverse(scene.CamInfo.Projection),
        scene.CamInfo.Position,
        0
    };
    cam->RBuffer[frame.FrameIndex]->CopyMapped(&camData, sizeof(CameraData));

    struct Data {
        int nRenderTarget;
        int nAccel;
        int nCameraBuffer;
        int nInstanceBuffer;
        int nSampler;
        int nCubeMap;
        int nFrameIndex;
        int nSamplesPerPixel;
        int nBouncesPerRay;
        glm::ivec3 Pad;
    } data = {
        out->Bindless(ViewType::Storage),
        scene.TopLevelAS->Bindless(),
        cam->Bindless(ViewType::None, frame.FrameIndex),
        scene.Resources.InstanceBuffer->SRV(),
        sampler->Bindless(),
        mSkybox->SkyboxCubeView->GetDescriptor().Index,
        mFrameIndex,
        mSamplesPerPixel,
        mBouncesPerRay
    };

    // Trace
    frame.CommandBuffer->BeginMarker("Trace Triangle");
    frame.CommandBuffer->Barrier(out->Texture, ResourceLayout::Storage);
    frame.CommandBuffer->SetViewport(0, 0, frame.Width, frame.Height);
    frame.CommandBuffer->SetRaytracingPipeline(mPipeline);
    frame.CommandBuffer->ComputePushConstants(&data, sizeof(data), 0);
    frame.CommandBuffer->TraceRays(frame.Width, frame.Height);
    frame.CommandBuffer->EndMarker();

    // Copy to backbuffer
    frame.CommandBuffer->BeginMarker("Copy To Backbuffer");
    frame.CommandBuffer->Barrier(out->Texture, ResourceLayout::CopySource);
    frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::CopyDest);
    frame.CommandBuffer->CopyTextureToTexture(frame.Backbuffer, out->Texture);
    frame.CommandBuffer->EndMarker();

    mFrameIndex++;
}

void MainPass::UI()
{
    ImGui::SliderInt("Samples Per Pixel", &mSamplesPerPixel, 1, 50);
    ImGui::SliderInt("Bounces Per Ray", &mBouncesPerRay, 1, 50);
}
