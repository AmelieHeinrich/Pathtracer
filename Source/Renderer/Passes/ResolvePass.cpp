//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-10 22:58:25
//

#include "ResolvePass.hpp"
#include "Renderer/RendererTools.hpp"
#include "Application.hpp"

ResolvePass::ResolvePass()
{
    ShaderFile file = ShaderCompiler::Load("Shaders/Resolve.hlsl");

    auto sig = std::make_shared<RootSignature>(std::vector<RootType>{RootType::PushConstant }, sizeof(glm::ivec4));
    mPipeline = std::make_shared<ComputePipeline>(file.Modules["CSMain"], sig);

    TextureDesc desc = {};
    desc.Width = WIDTH;
    desc.Height = HEIGHT;
    desc.Depth = 1;
    desc.Levels = 1;
    desc.Usage = TextureUsage::Storage | TextureUsage::ShaderResource | TextureUsage::RenderTarget;
    desc.Format = TextureFormat::RGBA16Float;
    desc.Name = "RT History";

    auto tex = RendererTools::CreateSharedTexture("RTHistory", desc);
    tex->AddView(ViewType::Storage, ViewDimension::Texture);
    tex->AddView(ViewType::ShaderResource, ViewDimension::Texture);
    tex->AddView(ViewType::RenderTarget, ViewDimension::Texture);
}

void ResolvePass::Render(Frame& frame, Scene& scene)
{
    auto out = RendererTools::Get("RTOutput");
    auto history = RendererTools::Get("RTHistory");

    struct Constants {
        int nCurrent;
        int nPrevious;
        int nFrameCount;
        int nPad;
    } data = {
        out->Bindless(ViewType::Storage),
        history->Bindless(ViewType::ShaderResource),
        frame.FrameCount,
        0
    };

    frame.CommandBuffer->BeginMarker("Resolve");

    // Resolve pass
    
    // Only do the resolve if there is something to resolve with...
    if (frame.FrameCount > 0) {
        frame.CommandBuffer->UAVBarrier(out->Texture);
        frame.CommandBuffer->Barrier(history->Texture, ResourceLayout::Shader);
        frame.CommandBuffer->SetComputePipeline(mPipeline);
        frame.CommandBuffer->ComputePushConstants(&data, sizeof(data), 0);
        frame.CommandBuffer->Dispatch(frame.Width / 7, frame.Height / 7, 1);
    }
    // Otherwise, clear the MF
    else {
        frame.CommandBuffer->Barrier(history->Texture, ResourceLayout::ColorWrite);
        frame.CommandBuffer->ClearRenderTarget(history->GetView(ViewType::RenderTarget), 0, 0, 0);
    }
    
    // Copy to history
    frame.CommandBuffer->Barrier(history->Texture, ResourceLayout::CopyDest);
    frame.CommandBuffer->Barrier(out->Texture, ResourceLayout::CopySource);
    frame.CommandBuffer->CopyTextureToTexture(history->Texture, out->Texture);
    frame.CommandBuffer->Barrier(history->Texture, ResourceLayout::Common);
    frame.CommandBuffer->Barrier(out->Texture, ResourceLayout::Common);

    frame.CommandBuffer->EndMarker();
}

void ResolvePass::UI()
{

}
