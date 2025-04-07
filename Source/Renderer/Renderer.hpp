//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:28:36
//

#pragma once

#include "Scene.hpp"
#include "Pass.hpp"

#include <memory>
#include <vector>

class Renderer
{
public:
    Renderer();
    ~Renderer();

    void Render(Frame& frame, Scene& scene);
    void UI();
private:
    std::vector<std::shared_ptr<Pass>> mPasses;
};
