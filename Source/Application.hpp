//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 13:55:41
//

#pragma once

#include <Oslo/Oslo.hpp>
#include <Oslo/RHI/Uploader.hpp>
#include <Oslo/RHI/BLAS.hpp>

class Application
{
public:
    Application();
    ~Application();
    
    void Run();
private:
    std::shared_ptr<Window> mWindow;
};
