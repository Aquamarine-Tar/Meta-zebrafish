#ifndef __FLUID_PARTICLE_VISUALIZER_H__
#define __FLUID_PARTICLE_VISUALIZER_H__

#include <deque>
#include <vector>
#include <Eigen/Core>

class PeridynoBridge;

namespace GUI
{
/**
 * FSI 流体粒子可视化（参考 Peridyno GLPointVisualNode 思路）:
 * - 批量 GL_POINTS 绘制
 * - 按速度模长 Jet 伪彩色映射（同 Peridyno ColorMapping::Jet）
 */
class FluidParticleVisualizer
{
public:
    enum class ColorMode
    {
        VelocityJet,   // 速度模长 -> Jet 色图
        Uniform        // 单色
    };

    FluidParticleVisualizer();

    void SetColorMode(ColorMode mode) { mColorMode = mode; }
    void SetPointSize(float size) { mPointSize = size; }
    void SetVelocityRange(float vmin, float vmax);
    void SetUniformColor(const Eigen::Vector4d& color) { mUniformColor = color; }

    // 从 PeridynoBridge 拉取最新粒子数据并更新颜色缓冲
    void Update(const PeridynoBridge* bridge);

    // 在已设置好 modelview 的坐标系下绘制（与虫体局部坐标一致）
    void Draw() const;

    int GetParticleCount() const { return mParticleCount; }
    float GetAverageSpeed() const { return mAverageSpeed; }
    float GetVelocityMax() const { return mVelocityMax; }
    const std::deque<float>& GetSpeedHistory() const { return mSpeedHistory; }
    void SetHistoryCapacity(int capacity) { mHistoryCapacity = std::max(2, capacity); }

private:
    void RecordSpeedSample();
    static float Clamp01(float x);
    static Eigen::Vector3f MapJetColor(float t);
    void UpdateColorsFromVelocity();

    ColorMode mColorMode;
    float mPointSize;
    float mVelocityMin;
    float mVelocityMax;
    Eigen::Vector4d mUniformColor;

    int mParticleCount;
    int mHistoryCapacity;
    float mAverageSpeed;
    std::deque<float> mSpeedHistory;
    std::vector<float> mPositions;
    std::vector<float> mVelocities;
    std::vector<float> mColors;
};
}

#endif
