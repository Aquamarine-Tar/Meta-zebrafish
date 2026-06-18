#!/usr/bin/env python3
import html
import os
import zipfile
from datetime import datetime, timezone

OUT = "PROJECT_REPORT_PPT.pptx"

W = 12192000
H = 6858000

BG = "F7FAFC"
INK = "0B172A"
MUTED = "475569"
BLUE = "2563EB"
CYAN = "0891B2"
TEAL = "0F766E"
GREEN = "16A34A"
ORANGE = "EA580C"
RED = "DC2626"
PURPLE = "7C3AED"
SLATE = "E2E8F0"
PANEL = "FFFFFF"
LINE = "CBD5E1"


def emu(x):
    return int(x * 914400)


def esc(s):
    return html.escape(str(s), quote=True)


def color_xml(rgb):
    return f'<a:solidFill><a:srgbClr val="{rgb}"/></a:solidFill>'


def nofill_xml():
    return '<a:noFill/>'


def ln_xml(rgb=LINE, width=12700):
    return f'<a:ln w="{width}"><a:solidFill><a:srgbClr val="{rgb}"/></a:solidFill></a:ln>'


def text_body(text, size=22, color=INK, bold=False, align="l", font="Microsoft YaHei", bullets=False, line_spacing=None):
    paras = str(text).split("\n") if text is not None else [""]
    out = ['<p:txBody><a:bodyPr wrap="square" rtlCol="0"/><a:lstStyle/>']
    for raw in paras:
        t = raw.strip()
        is_bullet = bullets or t.startswith("• ")
        if t.startswith("• "):
            t = t[2:].strip()
        ppr = f'<a:pPr algn="{align}"/>'
        if is_bullet:
            ppr = (
                f'<a:pPr marL="285750" indent="-171450" algn="{align}">'
                '<a:buFont typeface="Arial"/><a:buChar char="•"/></a:pPr>'
            )
        if line_spacing:
            ppr = ppr.replace("/>", f'><a:lnSpc><a:spcPct val="{line_spacing}"/></a:lnSpc></a:pPr>')
        b = ' b="1"' if bold else ""
        out.append(
            "<a:p>"
            + ppr
            + f'<a:r><a:rPr lang="zh-CN" sz="{int(size*100)}"{b}>'
            + f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            + f'<a:latin typeface="{esc(font)}"/><a:ea typeface="{esc(font)}"/></a:rPr>'
            + f"<a:t>{esc(t)}</a:t></a:r></a:p>"
        )
    out.append("</p:txBody>")
    return "".join(out)


def sp(shape_id, x, y, w, h, text="", fill=PANEL, line=LINE, preset="roundRect",
       size=20, color=INK, bold=False, align="ctr", name=None, bullets=False):
    name = name or f"Shape {shape_id}"
    fill_xml = nofill_xml() if fill is None else color_xml(fill)
    line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else ln_xml(line)
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="{preset}"><a:avLst/></a:prstGeom>
    {fill_xml}{line_xml}
  </p:spPr>
  {text_body(text, size=size, color=color, bold=bold, align=align, bullets=bullets)}
</p:sp>"""


def textbox(shape_id, x, y, w, h, text, size=20, color=INK, bold=False, align="l", bullets=False):
    return sp(shape_id, x, y, w, h, text, fill=None, line=None, preset="rect",
              size=size, color=color, bold=bold, align=align, bullets=bullets)


def line_shape(shape_id, x, y, w, h, rgb=LINE, width=20000):
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="Line {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
    <a:ln w="{width}"><a:solidFill><a:srgbClr val="{rgb}"/></a:solidFill><a:tailEnd type="triangle"/></a:ln>
  </p:spPr>
  <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
</p:sp>"""


def title_block(title, subtitle=None):
    parts = [textbox(10, 0.55, 0.28, 12.3, 0.55, title, size=28, color=INK, bold=True)]
    if subtitle:
        parts.append(textbox(11, 0.58, 0.86, 11.9, 0.28, subtitle, size=11.5, color=MUTED))
    parts.append(line_shape(12, 0.58, 1.18, 12.15, 0, rgb=SLATE, width=9000))
    return "".join(parts)


def footer(n):
    return (
        textbox(900, 0.55, 7.18, 4.4, 0.2, "Fish FSI Explicit Weak Coupling", size=8.5, color="64748B")
        + textbox(901, 12.25, 7.18, 0.5, 0.2, str(n), size=8.5, color="64748B", align="r")
    )


def slide_xml(content):
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr>{color_xml(BG)}</p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {content}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def bullet_card(shape_id, x, y, w, h, title, bullets, accent=BLUE):
    return (
        sp(shape_id, x, y, w, h, "", fill=PANEL, line=SLATE, preset="roundRect")
        + sp(shape_id + 1, x, y, 0.08, h, "", fill=accent, line=accent, preset="rect")
        + textbox(shape_id + 2, x + 0.22, y + 0.15, w - 0.4, 0.32, title, size=16, bold=True, color=INK)
        + textbox(shape_id + 3, x + 0.25, y + 0.58, w - 0.45, h - 0.7, "\n".join("• " + b for b in bullets), size=12.5, color=MUTED, bullets=False)
    )


def table(shape_id, x, y, w, h, headers, rows, col_widths=None, font_size=10.5):
    nrow = len(rows) + 1
    ncol = len(headers)
    if col_widths is None:
        col_widths = [w / ncol] * ncol
    row_h = h / nrow
    out = []
    yy = y
    sid = shape_id
    for r in range(nrow):
        xx = x
        for c in range(ncol):
            text = headers[c] if r == 0 else rows[r - 1][c]
            fill = "DBEAFE" if r == 0 else PANEL
            out.append(sp(sid, xx, yy, col_widths[c], row_h, text, fill=fill, line=SLATE,
                          preset="rect", size=font_size, color=INK if r == 0 else MUTED,
                          bold=(r == 0), align="ctr" if r == 0 else "l"))
            sid += 1
            xx += col_widths[c]
        yy += row_h
    return "".join(out)


def flow(nodes, x, y, box_w, box_h, gap, colors=None, start_id=100):
    out = []
    for i, node in enumerate(nodes):
        cx = x + i * (box_w + gap)
        color = colors[i] if colors else PANEL
        out.append(sp(start_id + i * 3, cx, y, box_w, box_h, node, fill=color, line=SLATE,
                      preset="roundRect", size=11.5, color=INK, bold=True))
        if i < len(nodes) - 1:
            out.append(sp(start_id + i * 3 + 1, cx + box_w + 0.05, y + box_h * 0.34, gap - 0.1, box_h * 0.32,
                          "→", fill=None, line=None, preset="rect", size=20, color=BLUE, bold=True))
    return "".join(out)


slides = []

slides.append(
    sp(20, 0, 0, 13.333, 7.5, "", fill="EAF2FF", line=None, preset="rect")
    + sp(21, 0.65, 0.65, 12.0, 5.9, "", fill=PANEL, line="BFDBFE", preset="roundRect")
    + textbox(22, 1.05, 1.35, 11.1, 0.65, "鱼体游动 FSI 显式弱耦合仿真", size=30, bold=True, color=INK, align="ctr")
    + textbox(23, 1.25, 2.15, 10.7, 0.55, "软体鱼 FEM · WCSPH 粒子流体 · VAP 近壁压力投影", size=18, color=BLUE, bold=True, align="ctr")
    + flow(["固体侧\nSoftCon / PD", "流体侧\nWCSPH", "近壁增强\nVAP band", "界面耦合\n压力→顶点力"],
           1.2, 3.25, 2.35, 0.82, 0.45, colors=["DBEAFE", "CCFBF1", "EDE9FE", "FFEDD5"], start_id=50)
    + textbox(90, 1.1, 5.45, 11.1, 0.5, "课程汇报 · 学术/科技风格方法说明", size=14, color=MUTED, align="ctr")
)

slides.append(title_block("汇报结构", "从力学问题到数值方法，再到实验诊断")
    + bullet_card(30, 0.75, 1.55, 3.7, 4.6, "问题与框架", [
        "鱼体游动是典型移动边界 FSI",
        "采用显式弱耦合而非强耦合迭代",
        "每步交换边界速度与表面力"], BLUE)
    + bullet_card(50, 4.82, 1.55, 3.7, 4.6, "核心算法", [
        "固体侧：四面体 FEM + Projective Dynamics",
        "驱动侧：肌肉主动收缩波",
        "流体侧：WCSPH + VAP 近壁压力投影"], TEAL)
    + bullet_card(70, 8.88, 1.55, 3.7, 4.6, "实验诊断", [
        "体积保持与四面体翻转",
        "表面压力覆盖与力矩",
        "headless 快照与粒子追踪"], ORANGE))

slides.append(title_block("研究问题：鱼体游动的 FSI", "内部肌肉驱动变形，外部流体反作用于鱼体")
    + sp(30, 0.8, 1.55, 5.6, 4.8, "", fill=PANEL, line=SLATE)
    + textbox(31, 1.05, 1.85, 5.1, 0.35, "连续力学视角", size=17, bold=True)
    + textbox(32, 1.1, 2.45, 4.9, 2.3,
              "固体区域 Ω_s(t):\nρ_s x¨ = div(P) + f_muscle + f_fluid\n\n流体区域 Ω_f(t):\nρ_f Dv/Dt = -∇p + μ∇²v\n\n界面 Γ(t):\nv_fluid · n = v_solid · n", size=13.5, color=INK)
    + sp(40, 7.0, 1.45, 5.5, 4.9, "", fill=PANEL, line=SLATE)
    + textbox(41, 7.25, 1.85, 5.0, 0.35, "本项目采用的离散思想", size=17, bold=True)
    + flow(["鱼体形状\nx_s, v_s", "流体边界\nΓ_s, u_Γ", "SPH/VAP\n求近壁压力", "FEM\n推进鱼体"],
           7.25, 2.7, 1.08, 0.85, 0.25, colors=["DBEAFE", "E0F2FE", "CCFBF1", "FFEDD5"], start_id=80)
    + textbox(90, 7.35, 4.55, 4.8, 0.85, "关键：不在每个时间步内反复迭代流固平衡，而是使用显式弱耦合交换数据。", size=14, color=MUTED))

slides.append(title_block("显式弱耦合主循环", "固体位置给流体，流体力再回传固体")
    + flow(["1 固体状态\nx_sⁿ, v_sⁿ", "2 提取表面\nΓ_sⁿ", "3 流体子步\nWCSPH/VAP", "4 表面压力\nF_fluidⁿ", "5 固体更新\nx_sⁿ⁺¹"],
           0.75, 2.0, 2.0, 0.95, 0.38, colors=["DBEAFE", "E0F2FE", "CCFBF1", "FDE68A", "FFEDD5"], start_id=30)
    + sp(60, 1.05, 4.15, 11.2, 1.35, "Fluid solve:  F_Γⁿ = WCSPH/VAP(Γ_sⁿ, u_Γⁿ, Δt_s)\nSolid solve:  M(x_sⁿ⁺¹ - x_sⁿ)/Δt_s² = internal_PD + muscle + F_Γⁿ",
         fill="F8FAFC", line=SLATE, preset="roundRect", size=16, color=INK, align="ctr")
    + textbox(80, 1.15, 5.9, 11.0, 0.45, "稳定化：高频固体步 + 流体子步 + 力截断 + 压力缩放", size=14, color=BLUE, bold=True, align="ctr"))

slides.append(title_block("总体系统架构", "模块分离：固体、流体、界面耦合与诊断")
    + sp(30, 0.85, 1.55, 3.25, 4.6, "固体侧\n\n四面体 FEM\nCorotated FEM\nProjective Dynamics\n肌肉主动约束", fill="DBEAFE", line="93C5FD", size=15, bold=True)
    + sp(40, 5.05, 1.55, 3.25, 4.6, "流体侧\n\nWCSPH 粒子流体\n均匀网格邻域\nVAP 近壁投影\nGPU 并行", fill="CCFBF1", line="5EEAD4", size=15, bold=True)
    + sp(50, 9.25, 1.55, 3.25, 4.6, "耦合与诊断\n\n表面压力积分\n顶点力回传\n体积/翻转监控\n快照与粒子追踪", fill="FFEDD5", line="FDBA74", size=15, bold=True)
    + line_shape(60, 4.15, 3.15, 0.85, 0, BLUE)
    + line_shape(61, 8.35, 3.15, 0.85, 0, BLUE))

slides.append(title_block("固体侧：鱼体离散模型", "被动四面体网格承载弹性变形")
    + table(30, 0.8, 1.65, 4.8, 2.1, ["项目", "数量/说明"],
            [["顶点", "7950"], ["四面体", "24676"], ["表面三角形", "由四面体外边界自动提取"], ["缩放", "fish.meta: 0.004"]],
            [1.55, 3.25], font_size=12)
    + sp(60, 6.2, 1.55, 5.9, 2.1, "fish.meta\n\nfish_body.mesh\nzf_nerve_cord.obj\nfish.sampling\n头尾与局部坐标参考点", fill="F8FAFC", line=SLATE, size=15, bold=True)
    + flow(["体网格\n被动材料", "外边界\n流体界面", "采样点\n状态/诊断"], 1.25, 4.55, 3.0, 0.85, 0.6,
           colors=["DBEAFE", "E0F2FE", "FDE68A"], start_id=90))

slides.append(title_block("被动单元与主动肌肉单元", "神经索/肌肉路径定义主动收缩的空间位置")
    + flow(["神经索/肌肉折线", "弧长采样\nmuscle segments", "寻找穿过或靠近\n的四面体", "添加\nLinearMuscleConstraint", "激活 a(t,s)\n改变主动收缩"],
           0.65, 2.0, 2.0, 0.95, 0.27, colors=["EDE9FE", "DBEAFE", "E0F2FE", "CCFBF1", "FFEDD5"], start_id=30)
    + bullet_card(70, 0.95, 4.25, 5.4, 1.65, "被动单元", [
        "来自 fish_body.mesh 的四面体 FEM",
        "描述组织弹性、质量、阻尼与大变形响应"], BLUE)
    + bullet_card(90, 7.0, 4.25, 5.4, 1.65, "主动单元", [
        "来自 zf_nerve_cord.obj 的折线路径",
        "按距离权重 w = exp(-|d|/0.005) 映射到四面体"], PURPLE))

slides.append(title_block("固体材料：Corotated FEM", "去除刚体旋转，只惩罚真实拉伸/剪切")
    + sp(30, 0.75, 1.5, 5.6, 4.9, "静止构型\nDm = [X1-X0, X2-X0, X3-X0]\n\n变形构型\nDs = [x1-x0, x2-x0, x3-x0]\n\n变形梯度\nF = Ds Dm⁻¹", fill=PANEL, line=SLATE, size=16, bold=True)
    + sp(50, 7.0, 1.5, 5.55, 4.9, "旋转分离\nF ≈ R S\n\n弹性能\nE ≈ μ||F-R||² + λ tr(RᵀF-I)²\n\n意义\n整体转动不产生虚假应力\n弯曲/压缩/拉伸产生恢复力", fill="F8FAFC", line=SLATE, size=16, bold=True)
    + textbox(80, 1.0, 6.35, 11.6, 0.35, "材料参数：E=2E5，局部硬区 E=4E5，ν=0.4，质量=10", size=13.5, color=BLUE, bold=True, align="ctr"))

slides.append(title_block("Projective Dynamics 求解", "局部投影 + 全局线性系统，稳定处理软体大变形")
    + flow(["惯性预测\nqₙ", "局部投影\npᵢ", "全局求解\nA x=b", "速度更新\nvₙ₊₁"],
           1.0, 1.85, 2.55, 0.95, 0.45, colors=["DBEAFE", "E0F2FE", "CCFBF1", "FFEDD5"], start_id=30)
    + sp(60, 1.0, 3.75, 11.4, 1.55, "qₙ = xₙ + Δt vₙ + Δt² M⁻¹ f_ext\n\n((1/Δt²)M + L)xₙ₊₁ = (1/Δt²)M qₙ + Jd",
         fill=PANEL, line=SLATE, preset="roundRect", size=20, color=INK, bold=True, align="ctr")
    + textbox(90, 1.15, 5.85, 11.1, 0.38, "当前设置：960 Hz，PD 最大迭代 50，速度阻尼 0.9999", size=13.5, color=MUTED, align="ctr"))

slides.append(title_block("肌肉驱动：沿身体传播的主动收缩波", "左右侧相位差产生鱼体弯曲波")
    + sp(30, 0.8, 1.55, 5.85, 2.45, "a(s,t) = offset + A sin(2πft + sign·2πks + φ_side)\n\nA = 0.25\nfrequency = 1.0 Hz\nspatial cycles = 1.0\n左右相位差 = π", fill=PANEL, line=SLATE, size=17, bold=True)
    + sp(60, 7.05, 1.55, 5.45, 2.45, "力学作用\n\n肌肉激活不是表面外力\n而是内部主动约束\n\n主动应变 → 身体弯曲 → 推动流体", fill="FFEDD5", line="FDBA74", size=17, bold=True)
    + flow(["头部", "波峰", "波谷", "尾部"], 1.7, 4.85, 1.8, 0.75, 0.55,
           colors=["DBEAFE", "E0F2FE", "CCFBF1", "FDE68A"], start_id=90)
    + textbox(120, 1.2, 6.1, 11.2, 0.35, "空间相位沿身体长度变化，时间相位随仿真推进。", size=13.5, color=MUTED, align="ctr"))

slides.append(title_block("为什么使用 SPH 粒子流体", "移动边界和大变形界面更容易处理")
    + table(30, 0.65, 1.45, 12.1, 4.65, ["方面", "传统网格方法", "SPH 粒子方法"],
            [["移动边界", "浸入边界/ALE/重剖分", "鱼体表面直接影响附近粒子"],
             ["自由表面/拓扑", "需界面追踪或重构", "粒子天然跟随界面"],
             ["压力精度", "压力场通常更平滑", "近壁低分辨率时更噪声"],
             ["计算特点", "网格/稀疏线性系统", "邻域搜索高度并行，适合 GPU"],
             ["本项目取舍", "边界精度强但复杂", "更适合快速实现大变形 FSI"]],
            [1.8, 5.0, 5.3], font_size=10.8)
    + textbox(80, 1.0, 6.35, 11.4, 0.35, "策略：用 WCSPH 处理主体流体，用 VAP 专门补强近壁压力。", size=14, color=BLUE, bold=True, align="ctr"))

slides.append(title_block("WCSPH：弱可压缩 SPH", "密度波动通过 Tait 状态方程转成压力")
    + sp(30, 0.8, 1.55, 5.7, 4.75, "核插值\nA(xᵢ) ≈ Σⱼ mⱼ Aⱼ/ρⱼ · W(|xᵢ-xⱼ|,h)\n\n密度\nρᵢ = Σⱼ mⱼ Wᵢⱼ\nρᵢ = max(ρᵢ,ρ₀)\n\n压力\npᵢ = ρ₀ c_s²/7 · ((ρᵢ/ρ₀)^7 - 1)", fill=PANEL, line=SLATE, size=15.5, bold=True)
    + sp(70, 7.0, 1.55, 5.55, 4.75, "SPH 力\n\n压力力\nfᵖᵢ = -Σⱼ mⱼ(pᵢ+pⱼ)/(2ρⱼ)∇Wᵢⱼ\n\n粘性力\nfᵛᵢ = Σⱼ mⱼ μ(vⱼ-vᵢ)/ρⱼ ∇²Wᵢⱼ\n\n参数：dx=0.011, h=2dx, ρ₀=1000, c_s=20", fill="F8FAFC", line=SLATE, size=14.5, bold=True))

slides.append(title_block("VAP：近壁压力投影", "在鱼体表面附近构造局部投影系统 A p = b")
    + bullet_card(30, 0.75, 1.4, 3.6, 4.9, "为什么需要 VAP", [
        "WCSPH 近壁邻域不完整",
        "表面压力覆盖可能稀疏",
        "ghost 过强或过弱都会影响稳定"], RED)
    + sp(55, 4.85, 1.4, 3.65, 4.9, "核心系统\n\n∇²p = ρ/Δt · ∇·u*\n\n离散为\nA p = b\n\np: 投影压力\nb: 散度/边界/密度源项\nA: 邻域权重算子", fill="EDE9FE", line="C4B5FD", size=16, bold=True)
    + bullet_card(80, 8.95, 1.4, 3.6, 4.9, "压力用途", [
        "修正 band 粒子速度",
        "补偿近壁不可压条件",
        "积分得到鱼体表面水动力"], PURPLE))

slides.append(title_block("VAP 的两个特别设计", "只在关键区域投影，并用 ghost 表达运动边界")
    + sp(30, 0.8, 1.45, 5.7, 4.95, "设计一：band 内外混合\n\nband 内：VAP 压力投影\nband 外：WCSPH 压力 + 粘性\n\n动机\n近壁压力最影响 FSI，不必全域投影\n\n计算集中在鱼体附近", fill="DBEAFE", line="93C5FD", size=16, bold=True)
    + sp(60, 6.85, 1.45, 5.7, 4.95, "设计二：ghost 边界粒子\n\nx_ghost = x_surface\nv_ghost = v_surface\n\n作用\n表达 u_fluid·n ≈ u_solid·n\n\n不作为压力自由度，只提供边界约束", fill="CCFBF1", line="5EEAD4", size=16, bold=True))

slides.append(title_block("VAP 子步完整流程", "从 band 标记到表面水压力积分")
    + flow(["1 band\n标记", "2 merged\n集合", "3 邻居表", "4 alpha", "5 aii\nsurface"],
           0.55, 1.55, 2.0, 0.78, 0.33, colors=["DBEAFE", "E0F2FE", "CCFBF1", "FDE68A", "FFEDD5"], start_id=30)
    + flow(["6 b 源项", "7 CG\n求 p", "8 速度\n修正", "9 压力\n缩放", "10 表面\n水压力"],
           0.55, 3.0, 2.0, 0.78, 0.33, colors=["DBEAFE", "E0F2FE", "CCFBF1", "FDE68A", "FFEDD5"], start_id=80)
    + sp(130, 1.05, 4.65, 11.2, 1.2, "关键公式\nαᵢ = Σⱼ W(||xᵢ-xⱼ||,h)       (Ap)ᵢ = aiiᵢpᵢ + Σⱼaᵢⱼpⱼ       F_face ≈ -p_face n_face A_face",
         fill=PANEL, line=SLATE, size=14.5, bold=True, align="ctr")
    + textbox(150, 1.1, 6.2, 11.0, 0.35, "VAP_PRESSURE_TO_HYDRO=0.75, VAP_CG_MAX_ITER=50, VAP_CG_TOL=0.1", size=12.5, color=MUTED, align="ctr"))

slides.append(title_block("FSI 界面力传递", "压力积分到三角面，再分配到 FEM 顶点")
    + flow(["VAP/WCSPH\n近壁压力", "三角面\np,n,A", "面力\nF=-p n A", "重心坐标\nu,v,w", "顶点外力\nF_i"],
           0.8, 1.75, 2.0, 0.9, 0.32, colors=["CCFBF1", "DBEAFE", "FDE68A", "FFEDD5", "EDE9FE"], start_id=30)
    + sp(70, 1.05, 4.15, 11.15, 1.45, "F_i0 += u F_face      F_i1 += v F_face      F_i2 += w F_face\n\n最终进入 FEM 的同一套顶点自由度，作为外力推进固体。", fill=PANEL, line=SLATE, size=18, bold=True, align="ctr")
    + textbox(100, 1.15, 6.0, 11.0, 0.35, "当前默认关闭 contact repulsion，主要观察 hydro pressure 贡献。", size=13.2, color=ORANGE, bold=True, align="ctr"))

slides.append(title_block("主要参数", "时间步、材料、流体与 VAP 参数")
    + table(30, 0.6, 1.35, 5.9, 5.0, ["类别", "参数", "当前值"],
            [["耦合", "固体频率", "960 Hz"], ["耦合", "流体子步", "18"], ["耦合", "max vertex force", "5 N"],
             ["固体", "杨氏模量", "2E5 / 4E5"], ["固体", "泊松比", "0.4"], ["固体", "肌肉刚度", "5E5"]],
            [1.25, 2.7, 1.95], font_size=10.8)
    + table(70, 6.85, 1.35, 5.9, 5.0, ["类别", "参数", "当前值"],
            [["流体", "粒子间距", "0.011"], ["流体", "参考密度", "1000"], ["流体", "声速", "20"],
             ["VAP", "band 半径", "2.5h"], ["VAP", "CG 迭代/容差", "50 / 0.1"], ["VAP", "ghost scale", "0.02 / 0.05"]],
            [1.25, 2.7, 1.95], font_size=10.8))

slides.append(title_block("诊断指标与实验入口", "不只看动画，还检查数值稳定性与物理量")
    + bullet_card(30, 0.75, 1.4, 3.7, 4.85, "固体稳定性", [
        "volume_ratio 接近 1",
        "min/max tet ratio",
        "inverted_tets = 0 更理想"], BLUE)
    + bullet_card(50, 4.82, 1.4, 3.7, 4.85, "流体/界面", [
        "表面水压力非零比例",
        "contact/hydro/total 分轨",
        "绕质心力矩与合力"], TEAL)
    + bullet_card(70, 8.88, 1.4, 3.7, 4.85, "运行入口", [
        "hydro_30frame: 快速稳定性",
        "fsi_force_snapshot: 顶点力快照",
        "vap_particle_trace: 近壁粒子追踪"], ORANGE))

slides.append(title_block("可视化与输出", "适合汇报展示的结果形式")
    + sp(30, 0.85, 1.45, 3.65, 4.95, "实时可视化\n\n鱼体表面\n四面体网格\n肌肉线\n轨迹/坐标\n流体粒子", fill="DBEAFE", line="93C5FD", size=16, bold=True)
    + sp(50, 4.85, 1.45, 3.65, 4.95, "力场快照\n\ncontact\nhydro\ntotal\nfaces\npositions\n力矩", fill="CCFBF1", line="5EEAD4", size=16, bold=True)
    + sp(70, 8.85, 1.45, 3.65, 4.95, "后处理图\n\n压力分布\n无偏位移\nVAP band 诊断\n粒子轨迹\n参数扫表", fill="FFEDD5", line="FDBA74", size=16, bold=True)
    + textbox(100, 1.0, 6.4, 11.4, 0.3, "建议演示顺序：正弦肌肉摆动 → 表面压力快照 → 体积/翻转诊断 → VAP 粒子追踪。", size=13, color=MUTED, align="ctr"))

slides.append(title_block("当前亮点与限制", "完整链路已经搭建，后续重点在物理标定和强耦合")
    + bullet_card(30, 0.75, 1.35, 5.75, 5.1, "亮点", [
        "完整鱼体 FSI 显式弱耦合链路",
        "固体侧能处理软体大变形和主动肌肉驱动",
        "流体侧 GPU WCSPH，近壁 VAP 改善压力覆盖",
        "提供压力、力矩、体积、翻转和粒子级诊断"], GREEN)
    + bullet_card(60, 6.85, 1.35, 5.75, 5.1, "限制与后续", [
        "当前不是强耦合，界面没有每步迭代平衡",
        "材料、肌肉与流体参数仍需实验标定",
        "SPH 分辨率有限，近壁仍依赖 ghost/VAP 参数",
        "后续可加入半隐式 FSI、推进效率和涡结构分析"], RED))

slides.append(title_block("总结", "鱼体游动 FSI 的显式弱耦合仿真方法")
    + sp(30, 0.95, 1.5, 11.45, 3.3, "本项目实现了：\n\n软体鱼 FEM + 肌肉主动波驱动 + WCSPH 粒子流体 + VAP 近壁压力投影 + 表面压力回传。\n\n核心思想是把复杂的强耦合问题拆成可验证、可诊断的显式弱耦合流程。", fill=PANEL, line=SLATE, size=21, bold=True, align="ctr")
    + flow(["固体大变形", "近壁压力", "界面受力", "稳定诊断"], 1.4, 5.35, 2.25, 0.8, 0.55,
           colors=["DBEAFE", "CCFBF1", "FFEDD5", "EDE9FE"], start_id=80))


def presentation_xml(n):
    ids = "\n".join(f'<p:sldId id="{256+i}" r:id="rId{i+1}"/>' for i in range(n))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
 <p:sldIdLst>{ids}</p:sldIdLst>
 <p:sldSz cx="{W}" cy="{H}" type="wide"/>
 <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''


def presentation_rels(n):
    rels = [
        f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>'
        for i in range(n)
    ]
    rels.append(f'<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(rels) + "</Relationships>"


def content_types(n):
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(n):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
''' + "".join(overrides) + "</Types>"


ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

THEME = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Academic Tech">
<a:themeElements>
<a:clrScheme name="Academic Tech">
<a:dk1><a:srgbClr val="0B172A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="F7FAFC"/></a:lt2>
<a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="0F766E"/></a:accent2>
<a:accent3><a:srgbClr val="EA580C"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4>
<a:accent5><a:srgbClr val="0891B2"/></a:accent5><a:accent6><a:srgbClr val="16A34A"/></a:accent6>
<a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
</a:clrScheme>
<a:fontScheme name="Microsoft YaHei"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:minorFont></a:fontScheme>
<a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle/></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
</a:themeElements></a:theme>'''


def core_props():
    now = datetime.now(timezone.utc).isoformat()
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>鱼体游动 FSI 显式弱耦合仿真</dc:title><dc:creator>Codex</dc:creator>
<cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'''


APP_PROPS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<Application>Codex OOXML Generator</Application><PresentationFormat>Widescreen</PresentationFormat>
<Slides>20</Slides></Properties>'''


def write_pptx():
    n = len(slides)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(n))
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("ppt/presentation.xml", presentation_xml(n))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(n))
        z.writestr("ppt/theme/theme1.xml", THEME)
        z.writestr("docProps/core.xml", core_props())
        z.writestr("docProps/app.xml", APP_PROPS.replace("<Slides>20</Slides>", f"<Slides>{n}</Slides>"))
        for i, content in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(content + footer(i)))
    return os.path.abspath(OUT), n


if __name__ == "__main__":
    path, n = write_pptx()
    print(f"wrote {path} with {n} slides")
