//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 13:56:21
//

#include "Application.hpp"

Application::Application()
{
    mStart = SDL_GetTicks();

    mWindow = std::make_shared<Window>(1280, 720, "Hello, World!");
    Oslo::AttachWindow(mWindow);

    mRenderer = std::make_shared<Renderer>();

    mScene.PushEntity(glm::mat4(1.0f), "Assets/Sponza/Sponza.gltf");
    mScene.Build();

    Uploader::Flush();
}

Application::~Application()
{

}
    
void Application::Run()
{
    while (mWindow->IsOpen()) {
        // Calculate DT
        float now = SDL_GetTicks();
        float dt = (now - mStart) / 1000.0f;
        mStart = now;

        // Begin camera
        mCamera.Begin();
        mScene.CamInfo.Position = mCamera.Position();
        mScene.CamInfo.View = mCamera.View();
        mScene.CamInfo.Projection = mCamera.Projection();

        // Begin frame
        mWindow->Update();
        Frame frame = RHI::Begin();
        frame.CommandBuffer->Begin();

        // Render
        mRenderer->Render(frame, mScene);

        // UI
        frame.CommandBuffer->BeginMarker("ImGui");
        frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::ColorWrite);
        frame.CommandBuffer->SetRenderTargets({ frame.BackbufferView }, nullptr);
        frame.CommandBuffer->BeginGUI(frame.Width, frame.Height);
        mRenderer->UI();
        frame.CommandBuffer->EndGUI();
        frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::Present);
        frame.CommandBuffer->EndMarker();
    
        // End frame
        frame.CommandBuffer->End();
        RHI::Submit({ frame.CommandBuffer });
        RHI::End();
        RHI::Present(false);
    
        // Update camera
        mCamera.Update(dt, frame.Width, frame.Height);
    }
    RHI::Wait();
}
