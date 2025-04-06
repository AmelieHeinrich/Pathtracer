//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-06 17:21:14
//

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
        specs.Signature = std::make_shared<RootSignature>(std::vector<RootType>{ RootType::PushConstant, RootType::ShaderResource }, sizeof(glm::ivec4));
        
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
        uint32_t indices[] = {
            0, 1, 2
        };

        glm::vec3 vertices[] = {
            glm::vec3{  0.0f, -0.5f, 1.0f },
            glm::vec3{ -0.5f,  0.5f, 1.0f },
            glm::vec3{  0.5f,  0.5f, 1.0f }
        };

        std::shared_ptr<Buffer> vertexBuffer = std::make_shared<Buffer>(sizeof(vertices), sizeof(glm::vec3), BufferType::Vertex, "Vertex Buffer");
        std::shared_ptr<Buffer> indexBuffer = std::make_shared<Buffer>(sizeof(indices), sizeof(uint32_t), BufferType::Index, "Index Buffer");

        Uploader::EnqueueBufferUpload(vertices, sizeof(vertices), vertexBuffer);
        Uploader::EnqueueBufferUpload(indices, sizeof(indices), indexBuffer);
        
        // Create acceleration structures
        std::shared_ptr<BLAS> blas = std::make_shared<BLAS>(vertexBuffer, indexBuffer, 3, 3, "BLAS");
        Uploader::EnqueueAccelerationStructureBuild(blas);

        RaytracingInstance instance = {};
        instance.Transform = glm::identity<glm::mat3x4>();
        instance.AccelerationStructure = blas->GetAddress();
        instance.InstanceMask = 1;
        instance.InstanceID = 0;
        instance.Flags = 0;

        std::shared_ptr<Buffer> instanceBuffer = std::make_shared<Buffer>(sizeof(RaytracingInstance), sizeof(RaytracingInstance), BufferType::Constant, "Scene Instances");
        instanceBuffer->CopyMapped(&instance, sizeof(instance));

        std::shared_ptr<TLAS> tlas = std::make_shared<TLAS>(instanceBuffer, 1, "Scene TLAS");
        Uploader::EnqueueAccelerationStructureBuild(tlas);
        
        // Flush before starting
        Uploader::Flush();
    
        // Main loop
        while (window->IsOpen()) {
            // Begin frame
            window->Update();
            Frame frame = RHI::Begin();
            frame.CommandBuffer->Begin();

            // Draw triangle
            struct Data {
                // Resources
                int nRenderTarget;
                glm::ivec3 nPad;
            } data = {
                rtUAV->GetDescriptor().Index,
                glm::ivec3(0)
            };

            // Trace rays
            frame.CommandBuffer->BeginMarker("Trace Triangle");
            frame.CommandBuffer->Barrier(rtOutput, ResourceLayout::Storage);
            frame.CommandBuffer->SetViewport(0, 0, frame.Width, frame.Height);
            frame.CommandBuffer->SetRaytracingPipeline(pipeline);
            frame.CommandBuffer->ComputePushConstants(&data, sizeof(data), 0);
            frame.CommandBuffer->BindComputeTLAS(tlas, 1);
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
            ImGui::ShowDemoWindow();
            frame.CommandBuffer->EndGUI();
            frame.CommandBuffer->Barrier(frame.Backbuffer, ResourceLayout::Present);
            frame.CommandBuffer->EndMarker();

            // End frame
            frame.CommandBuffer->End();
            RHI::Submit({ frame.CommandBuffer });
            RHI::End();
            RHI::Present(false);
        }
        RHI::Wait();
    }
    
    // Cleanup
    Oslo::Exit();
}
