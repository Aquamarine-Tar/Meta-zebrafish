#include "FluidParticleVisualizer.h"
#include "../sim/fluid/PeridynoBridge.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include <algorithm>
#include <cmath>

namespace GUI
{
FluidParticleVisualizer::FluidParticleVisualizer()
    : mColorMode(ColorMode::VelocityJet),
      mPointSize(6.0f),
      mVelocityMin(0.0f),
      mVelocityMax(0.15f),
      mUniformColor(0.2, 0.55, 0.95, 0.85),
      mParticleCount(0),
      mHistoryCapacity(80),
      mAverageSpeed(0.0f)
{
}

void FluidParticleVisualizer::RecordSpeedSample()
{
    mAverageSpeed = 0.0f;
    if (mParticleCount > 0 && !mVelocities.empty())
    {
        float sum = 0.0f;
        for (int i = 0; i < mParticleCount; ++i)
        {
            const float vx = mVelocities[3 * i + 0];
            const float vy = mVelocities[3 * i + 1];
            const float vz = mVelocities[3 * i + 2];
            sum += std::sqrt(vx * vx + vy * vy + vz * vz);
        }
        mAverageSpeed = sum / mParticleCount;
    }

    mSpeedHistory.push_back(mAverageSpeed);
    while ((int)mSpeedHistory.size() > mHistoryCapacity)
        mSpeedHistory.pop_front();
}

void FluidParticleVisualizer::SetVelocityRange(float vmin, float vmax)
{
    mVelocityMin = vmin;
    mVelocityMax = std::max(vmax, vmin + 1e-6f);
}

float FluidParticleVisualizer::Clamp01(float x)
{
    return std::max(0.0f, std::min(1.0f, x));
}

Eigen::Vector3f FluidParticleVisualizer::MapJetColor(float t)
{
    // 与 Peridyno ColorMapping::CM_MapJetColor 相同的分段三角波 Jet 映射
    const float x = Clamp01(t);
    const float r = Clamp01(-4.0f * std::fabs(x - 0.75f) + 1.5f);
    const float g = Clamp01(-4.0f * std::fabs(x - 0.50f) + 1.5f);
    const float b = Clamp01(-4.0f * std::fabs(x - 0.25f) + 1.5f);
    return Eigen::Vector3f(r, g, b);
}

void FluidParticleVisualizer::UpdateColorsFromVelocity()
{
    mColors.resize(mParticleCount * 3);
    if (mParticleCount == 0)
        return;

    if (mColorMode == ColorMode::Uniform)
    {
        for (int i = 0; i < mParticleCount; ++i)
        {
            mColors[3 * i + 0] = (float)mUniformColor[0];
            mColors[3 * i + 1] = (float)mUniformColor[1];
            mColors[3 * i + 2] = (float)mUniformColor[2];
        }
        return;
    }

    float vmax = mVelocityMax;
    if (mColorMode == ColorMode::VelocityJet)
    {
        float observedMax = 0.0f;
        for (int i = 0; i < mParticleCount; ++i)
        {
            const float vx = mVelocities[3 * i + 0];
            const float vy = mVelocities[3 * i + 1];
            const float vz = mVelocities[3 * i + 2];
            observedMax = std::max(observedMax, std::sqrt(vx * vx + vy * vy + vz * vz));
        }
        vmax = std::max(mVelocityMax, observedMax);
    }

    const float vmin = mVelocityMin;
    const float invRange = 1.0f / std::max(vmax - vmin, 1e-6f);

    for (int i = 0; i < mParticleCount; ++i)
    {
        const float vx = mVelocities[3 * i + 0];
        const float vy = mVelocities[3 * i + 1];
        const float vz = mVelocities[3 * i + 2];
        const float speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        const Eigen::Vector3f rgb = MapJetColor((speed - vmin) * invRange);
        mColors[3 * i + 0] = rgb[0];
        mColors[3 * i + 1] = rgb[1];
        mColors[3 * i + 2] = rgb[2];
    }
}

void FluidParticleVisualizer::Update(const PeridynoBridge* bridge)
{
    mParticleCount = 0;
    mPositions.clear();
    mVelocities.clear();
    mColors.clear();

    if (!bridge || !bridge->IsInitialized())
    {
        RecordSpeedSample();
        return;
    }

    bridge->GetFluidParticles(mPositions);
    bridge->GetFluidVelocities(mVelocities);
    mParticleCount = bridge->GetFluidParticleCount();

    if (mParticleCount <= 0)
    {
        RecordSpeedSample();
        return;
    }

    if (mPointSize <= 0.0f)
        mPointSize = std::max(4.0f, bridge->GetParticleSpacing() * 180.0f);

    UpdateColorsFromVelocity();
    RecordSpeedSample();
}

void FluidParticleVisualizer::Draw() const
{
    if (mParticleCount <= 0 || mPositions.empty())
        return;

    const bool lightingWasEnabled = glIsEnabled(GL_LIGHTING);
    if (lightingWasEnabled)
        glDisable(GL_LIGHTING);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glPointSize(mPointSize);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, mPositions.data());
    glColorPointer(3, GL_FLOAT, 0, mColors.data());
    glDrawArrays(GL_POINTS, 0, mParticleCount);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_BLEND);

    if (lightingWasEnabled)
        glEnable(GL_LIGHTING);
}
}
