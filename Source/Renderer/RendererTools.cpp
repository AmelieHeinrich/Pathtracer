//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:55:26
//

#include "RendererTools.hpp"

std::shared_ptr<View> RenderPassResource::AddView(ViewType type, ViewDimension dimension, TextureFormat format)
{
    std::shared_ptr<View> view = std::make_shared<View>(Texture, type, dimension, format);
    mViews.push_back(view);
    return view;
}

std::shared_ptr<View> RenderPassResource::GetView(ViewType type)
{
    for (auto& view : mViews) {
        if (view->GetType() == type) {
            return view;
        }
    }
    return nullptr;
}

int RenderPassResource::Bindless(ViewType type, int frameIndex)
{
    switch (Type) {
        case RenderPassResourceType::SharedTexture: {
            return GetView(type)->GetDescriptor().Index;
        }
        case RenderPassResourceType::SharedBuffer: {
            switch (type) {
                case ViewType::Storage: {
                    return Buffer->UAV();
                }
                case ViewType::ShaderResource: {
                    return Buffer->SRV();
                }
            }
            break;
        }
        case RenderPassResourceType::SharedSampler: {
            return Sampler->BindlesssSampler();
            break;
        }
        case RenderPassResourceType::SharedRingBuffer: {
            switch (type) {
                case ViewType::ShaderResource: {
                    return RBuffer[frameIndex]->SRV();
                }
                case ViewType::None: {
                    return RBuffer[frameIndex]->CBV();
                }
            }
        }
    }
    return -1;
}

RendererTools::Data RendererTools::sData;

void RendererTools::Init()
{
    {
        TextureDesc desc = {};
        desc.Width = 1;
        desc.Height = 1;
        desc.Levels = 1;
        desc.Depth = 1;
        desc.Name = "White Texture";
        desc.Format = TextureFormat::RGBA8;
        
        auto tex = CreateSharedTexture("WhiteTexture", desc);
        tex->AddView(ViewType::ShaderResource, ViewDimension::Texture);

        Uploader::EnqueueTextureUpload(std::vector<uint8_t>{ 0xFF, 0xFF, 0xFF, 0xFF }, tex->Texture);
    }
    {
        TextureDesc desc = {};
        desc.Width = 1;
        desc.Height = 1;
        desc.Levels = 1;
        desc.Depth = 1;
        desc.Name = "Black Texture";
        desc.Format = TextureFormat::RGBA8;
        
        auto tex = CreateSharedTexture("BlackTexture", desc);
        tex->AddView(ViewType::ShaderResource, ViewDimension::Texture);
        
        Uploader::EnqueueTextureUpload(std::vector<uint8_t>{ 0x00, 0x00, 0x00, 0xFF }, tex->Texture);
    }
}

void RendererTools::Free()
{
    sData.Resources.clear();
}

std::shared_ptr<RenderPassResource> RendererTools::CreateSharedTexture(const std::string& name, TextureDesc desc)
{
    std::shared_ptr<RenderPassResource> resource = std::make_shared<RenderPassResource>();
    resource->Type = RenderPassResourceType::SharedTexture;
    resource->Texture = std::make_shared<Texture>(desc);
    sData.Resources[name] = resource;
    return sData.Resources[name];
}

std::shared_ptr<RenderPassResource> RendererTools::CreateSharedRingBuffer(const std::string& name, uint64_t size, uint64_t stride)
{
    std::shared_ptr<RenderPassResource> resource = std::make_shared<RenderPassResource>();
    resource->Type = RenderPassResourceType::SharedRingBuffer;
    for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
        resource->RBuffer[i] = std::make_shared<Buffer>(size, stride, BufferType::Constant, name + " " + std::to_string(i));
        resource->RBuffer[i]->BuildCBV();
        if (stride > 0)
            resource->RBuffer[i]->BuildSRV();
    }
    sData.Resources[name] = resource;
    return sData.Resources[name];
}

std::shared_ptr<RenderPassResource> RendererTools::CreateSharedRWBuffer(const std::string& name, uint64_t size, uint64_t stride)
{
    std::shared_ptr<RenderPassResource> resource = std::make_shared<RenderPassResource>();
    resource->Type = RenderPassResourceType::SharedBuffer;
    resource->Buffer = std::make_shared<Buffer>(size, stride, BufferType::Storage, name);
    resource->Buffer->BuildSRV();
    resource->Buffer->BuildUAV();
    sData.Resources[name] = resource;
    return sData.Resources[name];
}

std::shared_ptr<RenderPassResource> RendererTools::CreateSharedSampler(const std::string& name, SamplerFilter filter, SamplerAddress address, bool mips, bool comparison)
{
    std::shared_ptr<RenderPassResource> resource = std::make_shared<RenderPassResource>();
    resource->Type = RenderPassResourceType::SharedSampler;
    resource->Sampler = std::make_shared<Sampler>(address, filter, mips, 16, comparison);
    sData.Resources[name] = resource;
    return sData.Resources[name];
}

std::shared_ptr<RenderPassResource> RendererTools::Get(const std::string& name)
{
    return sData.Resources[name];
}
