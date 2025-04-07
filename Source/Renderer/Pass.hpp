//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 15:42:46
//

#pragma once

#include "Scene.hpp"

class Pass
{
public:
    virtual void Render(Frame& frame, Scene& scene) = 0;
    virtual void UI() = 0;
};
