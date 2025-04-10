//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 21:20:01
//

#include "TextureCache.hpp"

std::unordered_map<std::string, std::shared_ptr<Texture>> TextureCache::mTextures;

std::shared_ptr<Texture> TextureCache::Get(const std::string& path)
{
    if (mTextures.find(path) == mTextures.end()) {
        ImageData data;
        data.Load(path);

        TextureDesc desc;
        desc.Width = data.Width;
        desc.Height = data.Height;
        desc.Depth = 1;
        desc.Levels = 1;
        desc.Name = path;
        desc.Usage = TextureUsage::ShaderResource;
        desc.Format = TextureFormat::RGBA8_sRGB;
        
        mTextures[path] = std::make_shared<Texture>(desc);
        Uploader::EnqueueTextureUpload(data.Pixels, mTextures[path]);
        return mTextures[path];
    } else {
        return mTextures[path];
    }

    return nullptr;
}

void TextureCache::Clear()
{
    mTextures.clear();
}
