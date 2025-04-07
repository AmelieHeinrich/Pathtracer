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
