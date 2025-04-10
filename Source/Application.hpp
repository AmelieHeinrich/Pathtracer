//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 13:55:41
//

#pragma once

#include <Oslo/Oslo.hpp>

#include "Renderer/Renderer.hpp"
#include "Scene.hpp"
#include "Camera.hpp"

class Application
{
public:
    Application();
    ~Application();
    
    void Run();
private:
    void UI();

    std::shared_ptr<Window> mWindow;
    std::shared_ptr<Renderer> mRenderer;

    Scene mScene;
    Camera mCamera;

    float mStart = 0.0f;
    bool mInputCamera = true;
};
