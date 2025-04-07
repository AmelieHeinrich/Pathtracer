//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 21:18:57
//

#pragma once

#include <Oslo/Oslo.hpp>
#include <unordered_map>

class TextureCache
{
public:
    static std::shared_ptr<Texture> Get(const std::string& path);
    static void Clear();

private:
    static std::unordered_map<std::string, std::shared_ptr<Texture>> mTextures;
};
