//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 13:56:21
//

#include "Application.hpp"

Application::Application()
{
    mWindow = std::make_shared<Window>(1280, 720, "Hello, World!");
    Oslo::AttachWindow(mWindow);
}

Application::~Application()
{

}
    
void Application::Run()
{
    while (mWindow->IsOpen()) {
        mWindow->Update();
    }
}
