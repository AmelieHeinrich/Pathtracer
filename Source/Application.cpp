//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 13:56:21
//

#include "Application.hpp"

#include <imgui.h>

Application::Application()
{
    mStart = SDL_GetTicks();

    mWindow = std::make_shared<Window>(WIDTH, HEIGHT, "Hello, World!");
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
        UI();
        frame.CommandBuffer->EndGUI();
        frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::Present);
        frame.CommandBuffer->EndMarker();
    
        // End frame
        frame.CommandBuffer->End();
        RHI::Submit({ frame.CommandBuffer });
        RHI::End();
        RHI::Present(false);
    
        // Update camera
        if (mInputCamera) {
            mCamera.Update(dt, frame.Width, frame.Height);
        }
    }
    RHI::Wait();
}

void Application::UI()
{
    static bool p_open = true;

    ImGuiIO& io = ImGui::GetIO();
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoDocking;
    const float PAD = 10.0f;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 work_pos = viewport->WorkPos;
    ImVec2 work_size = viewport->WorkSize;
    ImVec2 window_pos, window_pos_pivot;
    window_pos.x = (work_pos.x + PAD);
    window_pos.y = (work_pos.y + PAD);
    window_pos_pivot.x = 0.0f;
    window_pos_pivot.y = 0.0f;
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    window_flags |= ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowBgAlpha(0.70f);
    ImGui::Begin("Example: Simple overlay", &p_open, window_flags);
    ImGui::Text("Pathtracer : a DXR pathtracer by Amélie Heinrich");
    ImGui::Text("GPU: %s", RHI::GetDevice()->GetDeviceName().c_str());
    ImGui::Separator();

    mRenderer->UI();

    mInputCamera = !ImGui::IsWindowFocused();

    ImGui::End();
}
