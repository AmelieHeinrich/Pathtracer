//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-06 11:14:48
//

#include <memory>

#include <Oslo/Oslo.hpp>
#include <Oslo/RHI/Uploader.hpp>
#include <Oslo/RHI/BLAS.hpp>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Camera.hpp"
#include "Model.hpp"
#include "Scene.hpp"

int main(void)
{
    // Initialize Oslo
    Oslo::Init();
    std::shared_ptr<Window> window = std::make_shared<Window>(1280, 720, "Hello, World!");
    Oslo::AttachWindow(window);

    {
        // Create pipeline
        ShaderFile file = ShaderCompiler::Load("Shaders/Simple.hlsl");

        RaytracingPipelineSpecs specs = {};
        specs.AttribSize = sizeof(glm::vec2);
        specs.MaxRecursion = 3;
        specs.PayloadSize = sizeof(glm::vec4);
        specs.Library = file.Modules["Shader"];
        specs.Signature = std::make_shared<RootSignature>(std::vector<RootType>{ RootType::PushConstant }, sizeof(glm::ivec4));
        
        std::shared_ptr<RaytracingPipeline> pipeline = std::make_shared<RaytracingPipeline>(specs);

        // Create output
        TextureDesc desc = {};
        desc.Width = 1280;
        desc.Height = 720;
        desc.Depth = 1;
        desc.Levels = 1;
        desc.Usage = TextureUsage::Storage;
        desc.Format = TextureFormat::RGBA8;
        desc.Name = "RT Output";

        std::shared_ptr<Texture> rtOutput = std::make_shared<Texture>(desc);
        std::shared_ptr<View> rtUAV = std::make_shared<View>(rtOutput, ViewType::Storage);

        // Create geometry
        Scene scene;
        scene.PushEntity(glm::mat4(1.0f), "Assets/Sponza/Sponza.gltf");
        scene.Build();
        
        // Flush before starting
        Uploader::Flush();

        // Create camera constant buffer
        std::shared_ptr<Buffer> cameraBuffer = std::make_shared<Buffer>(256, 0, BufferType::Constant, "Camera Constant Buffer");
        cameraBuffer->BuildCBV();

        // Setup timer and camera
        float start = SDL_GetTicks();
        Camera camera;

        // Main loop
        while (window->IsOpen()) {
            // Calculate dt
            float now = SDL_GetTicks();
            float dt = (now - start) / 1000.0f;
            start = now;

            // Begin frame
            window->Update();
            Frame frame = RHI::Begin();
            frame.CommandBuffer->Begin();
            camera.Begin();

            struct CameraData {
                glm::mat4 invView;
                glm::mat4 invProj;
                glm::vec3 cameraPos;
                int pad;
            } camData = {
                glm::inverse(camera.View()),
                glm::inverse(camera.Projection()),
                camera.Position(),
                0
            };
            cameraBuffer->CopyMapped(&camData, sizeof(CameraData));

            // Draw triangle
            struct Data {
                // Resources
                int nRenderTarget;
                int nAccel;
                int nCameraBuffer;
                int Pad;
            } data = {
                rtUAV->GetDescriptor().Index,
                scene.TopLevelAS->Bindless(),
                cameraBuffer->CBV(),
                0
            };

            // Trace rays
            frame.CommandBuffer->BeginMarker("Trace Triangle");
            frame.CommandBuffer->Barrier(rtOutput, ResourceLayout::Storage);
            frame.CommandBuffer->SetViewport(0, 0, frame.Width, frame.Height);
            frame.CommandBuffer->SetRaytracingPipeline(pipeline);
            frame.CommandBuffer->ComputePushConstants(&data, sizeof(data), 0);
            frame.CommandBuffer->TraceRays(frame.Width, frame.Height);
            frame.CommandBuffer->EndMarker();

            // Copy to backbuffer
            frame.CommandBuffer->BeginMarker("Copy To Backbuffer");
            frame.CommandBuffer->Barrier(rtOutput, ResourceLayout::CopySource);
            frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::CopyDest);
            frame.CommandBuffer->CopyTextureToTexture(frame.Backbuffer, rtOutput);
            frame.CommandBuffer->EndMarker();

            // Draw ImGui
            frame.CommandBuffer->BeginMarker("ImGui");
            frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::ColorWrite);
            frame.CommandBuffer->SetRenderTargets({ frame.BackbufferView }, nullptr);
            frame.CommandBuffer->BeginGUI(frame.Width, frame.Height);
            frame.CommandBuffer->EndGUI();
            frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::Present);
            frame.CommandBuffer->EndMarker();

            // End frame
            frame.CommandBuffer->End();
            RHI::Submit({ frame.CommandBuffer });
            RHI::End();
            RHI::Present(false);

            // Update camera
            camera.Update(dt, frame.Width, frame.Height);
        }
        RHI::Wait();
    }
    
    // Cleanup
    Oslo::Exit();
}
