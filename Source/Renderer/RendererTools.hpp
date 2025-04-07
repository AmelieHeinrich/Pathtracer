//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:51:07
//

#pragma once

#include <Oslo/Oslo.hpp>

#include <array>
#include <memory>
#include <vector>
#include <unordered_map>

using RingBuffer = std::array<std::shared_ptr<Buffer>, FRAMES_IN_FLIGHT>;

enum class RenderPassResourceType
{
    SharedTexture,
    SharedRingBuffer,
    SharedBuffer,
    SharedSampler
};

class RenderPassResource
{
public:
    RenderPassResourceType Type;
    std::shared_ptr<Texture> Texture;
    RingBuffer RBuffer;
    std::shared_ptr<Buffer> Buffer;
    std::shared_ptr<Sampler> Sampler;

    std::shared_ptr<View> AddView(ViewType type, ViewDimension dimension, TextureFormat format = TextureFormat::Unknown);
    std::shared_ptr<View> GetView(ViewType type);
    int Bindless(ViewType type = ViewType::None, int frameIndex = 0);
private:
    std::vector<std::shared_ptr<View>> mViews;

    friend class RendererTools;
};

class RendererTools
{
public:
    static void Init();
    static void Free();

    static std::shared_ptr<RenderPassResource> CreateSharedTexture(const std::string& name, TextureDesc desc);
    static std::shared_ptr<RenderPassResource> CreateSharedRingBuffer(const std::string& name, uint64_t size, uint64_t stride = 0);
    static std::shared_ptr<RenderPassResource> CreateSharedRWBuffer(const std::string& name, uint64_t size, uint64_t stride);
    static std::shared_ptr<RenderPassResource> CreateSharedSampler(const std::string& name, SamplerFilter filter, SamplerAddress address, bool mips = false, bool comparison = false);
    static std::shared_ptr<RenderPassResource> Get(const std::string& name);
private:
    static struct Data {
        std::unordered_map<std::string, std::shared_ptr<RenderPassResource>> Resources;
    } sData;
};
