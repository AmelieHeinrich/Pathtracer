//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-10 22:58:25
//

#include "TonemapPass.hpp"
#include "Renderer/RendererTools.hpp"
#include "Application.hpp"

#include <imgui.h>

TonemapPass::TonemapPass()
{
    ShaderFile file = ShaderCompiler::Load("Shaders/Tonemap.hlsl");

    auto sig = std::make_shared<RootSignature>(std::vector<RootType>{RootType::PushConstant }, sizeof(glm::ivec4));
    mPipeline = std::make_shared<ComputePipeline>(file.Modules["CSMain"], sig);

    TextureDesc desc = {};
    desc.Width = WIDTH;
    desc.Height = HEIGHT;
    desc.Depth = 1;
    desc.Levels = 1;
    desc.Usage = TextureUsage::Storage | TextureUsage::ShaderResource;
    desc.Format = TextureFormat::RGBA8;
    desc.Name = "LDR Output";

    auto tex = RendererTools::CreateSharedTexture("LDR", desc);
    tex->AddView(ViewType::Storage, ViewDimension::Texture);
    tex->AddView(ViewType::ShaderResource, ViewDimension::Texture);
}

void TonemapPass::Render(Frame& frame, Scene& scene)
{
    auto hdr = RendererTools::Get("RTOutput");
    auto ldr = RendererTools::Get("LDR");

    struct Constants {
        int nHDR;
        int nLDR;
        float fGamma;
        int nPad;
    } data = {
        hdr->Bindless(ViewType::ShaderResource),
        ldr->Bindless(ViewType::Storage),
        mGamma,
        0
    };

    frame.CommandBuffer->BeginMarker("Tonemap");

    // Tonemap
    frame.CommandBuffer->Barrier(hdr->Texture, ResourceLayout::Shader);
    frame.CommandBuffer->Barrier(ldr->Texture, ResourceLayout::Storage);
    frame.CommandBuffer->SetComputePipeline(mPipeline);
    frame.CommandBuffer->ComputePushConstants(&data, sizeof(data), 0);
    frame.CommandBuffer->Dispatch(frame.Width / 7, frame.Height / 7, 1);
    
    // Copy to backbuffer
    frame.CommandBuffer->Barrier(ldr->Texture, ResourceLayout::CopySource);
    frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::CopyDest);
    frame.CommandBuffer->CopyTextureToTexture(frame.Backbuffer, ldr->Texture);
    frame.CommandBuffer->Barrier(ldr->Texture, ResourceLayout::Common);
    frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::Common);
    
    frame.CommandBuffer->EndMarker();
}

void TonemapPass::UI()
{
    ImGui::SliderFloat("Gamma", &mGamma, 1.0f, 5.0f, "%.2f");
}
