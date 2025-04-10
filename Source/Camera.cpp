//
// > Notice: Amélie Heinrich @ 2025
// > Create Time: 2025-04-07 00:33:14
//

#include "Camera.hpp"

#include <imgui.h>

void Camera::Begin()
{
    ImVec2 pos = ImGui::GetMousePos();
    mLastX = pos.x;
    mLastY = pos.y;
}

void Camera::Update(float dt, int width, int height)
{
    if (ImGui::IsKeyDown(ImGuiKey_Z)) {
        RHI::ResetFrameCount();
        mPosition += mForward * dt * 3.0f;
    }
    if (ImGui::IsKeyDown(ImGuiKey_S)) {
        RHI::ResetFrameCount();
        mPosition -= mForward * dt * 3.0f;
    }
    if (ImGui::IsKeyDown(ImGuiKey_Q)) {
        RHI::ResetFrameCount();
        mPosition -= mRight * dt * 3.0f;
    }
    if (ImGui::IsKeyDown(ImGuiKey_D)) {
        RHI::ResetFrameCount();
        mPosition += mRight * dt * 3.0f;
    }

    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 pos = ImGui::GetMousePos();
        float dx = (pos.x - mLastX) * 0.1f;
        float dy = (pos.y - mLastY) * 0.1f;

        if (abs(dx) > 0.01f || abs(dy) > 0.01f) {
            RHI::ResetFrameCount();
        }

        mYaw += dx;
        mPitch -= dy;
    }

    mForward.x = glm::cos(glm::radians(mYaw)) * glm::cos(glm::radians(mPitch));
    mForward.y = glm::sin(glm::radians(mPitch));
    mForward.z = glm::sin(glm::radians(mYaw)) * glm::cos(glm::radians(mPitch));
    mForward = glm::normalize(mForward);

    mRight = glm::normalize(glm::cross(mForward, glm::vec3(0.0f, 1.0f, 0.0f)));
    mUp = glm::normalize(glm::cross(mRight, mForward));

    mView = glm::lookAt(mPosition, mPosition + mForward, glm::vec3(0.0f, 1.0f, 0.0f));
    mProjection = glm::perspective(glm::radians(90.0f), (float)width / (float)height, CAMERA_NEAR, CAMERA_FAR);
}
