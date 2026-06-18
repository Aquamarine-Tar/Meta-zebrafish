#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

import numpy as np


def read_medit_vertices(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        i = lines.index("Vertices")
    except ValueError as exc:
        raise ValueError(f"{path} does not contain a Vertices section") from exc

    n = int(lines[i + 1].strip())
    points = []
    for line in lines[i + 2 : i + 2 + n]:
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"Malformed vertex line: {line}")
        points.append([float(fields[0]), float(fields[1]), float(fields[2])])
    return np.asarray(points, dtype=float)


def compute_pca(points: np.ndarray):
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    # Keep the long-axis parameter increasing roughly with +Z for easier reading.
    if axis[2] < 0.0:
        axis = -axis
    s = centered @ axis
    return center, axis, s


def build_html(points: np.ndarray, center: np.ndarray, axis: np.ndarray, s: np.ndarray, source: Path) -> str:
    payload = {
        "source": str(source),
        "points": points[:, [0, 2]].round(6).tolist(),
        "xyz": points.round(6).tolist(),
        "center": center.round(8).tolist(),
        "axis": axis.round(10).tolist(),
        "sMin": float(s.min()),
        "sMax": float(s.max()),
        "bboxMin": points.min(axis=0).round(8).tolist(),
        "bboxMax": points.max(axis=0).round(8).tolist(),
    }
    data = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fish PCA Top View</title>
<style>
  html, body {{
    margin: 0;
    height: 100%;
    font-family: Arial, Helvetica, sans-serif;
    color: #16202a;
    background: #f6f7f9;
  }}
  #app {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    height: 100%;
  }}
  #canvas {{
    width: 100%;
    height: 100%;
    display: block;
    background: #ffffff;
    cursor: crosshair;
  }}
  aside {{
    border-left: 1px solid #d8dee8;
    padding: 16px;
    overflow: auto;
    background: #fbfcfe;
  }}
  h1 {{
    font-size: 18px;
    margin: 0 0 12px;
  }}
  h2 {{
    font-size: 14px;
    margin: 18px 0 8px;
  }}
  .kv {{
    font-size: 12px;
    line-height: 1.55;
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: #eef2f7;
    border: 1px solid #dbe2ec;
    padding: 10px;
    border-radius: 6px;
  }}
  button {{
    border: 1px solid #bdc7d6;
    background: #ffffff;
    color: #16202a;
    border-radius: 6px;
    padding: 7px 10px;
    margin: 6px 6px 0 0;
    cursor: pointer;
  }}
  button:hover {{ background: #edf3fb; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th, td {{
    border-bottom: 1px solid #e2e7ef;
    padding: 6px 4px;
    text-align: right;
  }}
  th:first-child, td:first-child {{ text-align: left; }}
  .hint {{
    font-size: 12px;
    color: #536170;
    line-height: 1.45;
  }}
</style>
</head>
<body>
<div id="app">
  <canvas id="canvas"></canvas>
  <aside>
    <h1>Fish PCA Top View (+Y)</h1>
    <div class="hint">
      View plane: X-Z. Drag to pan, mouse wheel to zoom, click to add a cut marker perpendicular to the PCA axis.
    </div>
    <h2>Mesh / PCA</h2>
    <div class="kv" id="info"></div>
    <button id="fit">Fit</button>
    <button id="clear">Clear Markers</button>
    <h2>Hover</h2>
    <div class="kv" id="hover">Move over the fish body.</div>
    <h2>Cut Markers</h2>
    <table>
      <thead><tr><th>#</th><th>u</th><th>s</th><th>x</th><th>z</th></tr></thead>
      <tbody id="markers"></tbody>
    </table>
  </aside>
</div>
<script>
const data = {data};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');
const hoverBox = document.getElementById('hover');
const markersBody = document.getElementById('markers');

let dpr = window.devicePixelRatio || 1;
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let last = null;
let markers = [];

const pts = data.points.map(p => ({{x:p[0], z:p[1]}}));
const cx = data.center[0];
const cz = data.center[2];
const ax = data.axis[0];
const az = data.axis[2];
const axisLen2 = ax * ax + az * az;
const sMin = data.sMin;
const sMax = data.sMax;

info.textContent =
`source: ${{data.source}}
vertices: ${{pts.length}}
bbox min xyz: ${{data.bboxMin.map(v => v.toFixed(4)).join(', ')}}
bbox max xyz: ${{data.bboxMax.map(v => v.toFixed(4)).join(', ')}}
center xyz: ${{data.center.map(v => v.toFixed(4)).join(', ')}}
PCA axis xyz: ${{data.axis.map(v => v.toFixed(5)).join(', ')}}
s range: ${{sMin.toFixed(4)}} .. ${{sMax.toFixed(4)}}`;

function resize() {{
  const rect = canvas.getBoundingClientRect();
  dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function fit() {{
  const rect = canvas.getBoundingClientRect();
  const xs = pts.map(p => p.x);
  const zs = pts.map(p => p.z);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minZ = Math.min(...zs), maxZ = Math.max(...zs);
  const pad = 36;
  scale = Math.min((rect.width - 2 * pad) / (maxX - minX), (rect.height - 2 * pad) / (maxZ - minZ));
  offsetX = pad - minX * scale;
  offsetY = rect.height - pad + minZ * scale;
  draw();
}}

function worldToScreen(x, z) {{
  return {{x: x * scale + offsetX, y: -z * scale + offsetY}};
}}

function screenToWorld(x, y) {{
  return {{x: (x - offsetX) / scale, z: -(y - offsetY) / scale}};
}}

function axisS(x, z) {{
  return ((x - cx) * ax + (z - cz) * az) / Math.sqrt(axisLen2);
}}

function axisUFromS(s) {{
  return (s - sMin) / (sMax - sMin);
}}

function axisPointFromS(s) {{
  const norm = Math.sqrt(axisLen2);
  return {{x: cx + (s / norm) * ax, z: cz + (s / norm) * az}};
}}

function drawGrid(rect) {{
  const stepWorld = niceStep(90 / scale);
  const topLeft = screenToWorld(0, 0);
  const bottomRight = screenToWorld(rect.width, rect.height);
  const minX = Math.floor(topLeft.x / stepWorld) * stepWorld;
  const maxX = Math.ceil(bottomRight.x / stepWorld) * stepWorld;
  const minZ = Math.floor(bottomRight.z / stepWorld) * stepWorld;
  const maxZ = Math.ceil(topLeft.z / stepWorld) * stepWorld;
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#e8edf4';
  ctx.fillStyle = '#667383';
  ctx.font = '11px Arial';
  for (let x = minX; x <= maxX; x += stepWorld) {{
    const a = worldToScreen(x, minZ), b = worldToScreen(x, maxZ);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.fillText(x.toFixed(1), a.x + 3, rect.height - 8);
  }}
  for (let z = minZ; z <= maxZ; z += stepWorld) {{
    const a = worldToScreen(minX, z), b = worldToScreen(maxX, z);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.fillText(z.toFixed(1), 8, a.y - 3);
  }}
}}

function niceStep(raw) {{
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const v = raw / pow;
  if (v < 2) return pow;
  if (v < 5) return 2 * pow;
  return 5 * pow;
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawGrid(rect);

  ctx.fillStyle = 'rgba(30, 100, 160, 0.45)';
  for (const p of pts) {{
    const q = worldToScreen(p.x, p.z);
    ctx.fillRect(q.x - 1.1, q.y - 1.1, 2.2, 2.2);
  }}

  const a = axisPointFromS(sMin);
  const b = axisPointFromS(sMax);
  const as = worldToScreen(a.x, a.z);
  const bs = worldToScreen(b.x, b.z);
  ctx.strokeStyle = '#d12d2d';
  ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(as.x, as.y); ctx.lineTo(bs.x, bs.y); ctx.stroke();
  ctx.fillStyle = '#d12d2d';
  ctx.font = '12px Arial';
  ctx.fillText('PCA axis  u=0', as.x + 6, as.y - 6);
  ctx.fillText('u=1', bs.x + 6, bs.y - 6);

  for (let i = 0; i < markers.length; i++) {{
    const m = markers[i];
    const p = axisPointFromS(m.s);
    const nx = -az / Math.sqrt(axisLen2);
    const nz = ax / Math.sqrt(axisLen2);
    const len = 300;
    const p0 = worldToScreen(p.x - nx * len, p.z - nz * len);
    const p1 = worldToScreen(p.x + nx * len, p.z + nz * len);
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 5]);
    ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
    ctx.setLineDash([]);
    const ps = worldToScreen(p.x, p.z);
    ctx.fillStyle = '#111827';
    ctx.beginPath(); ctx.arc(ps.x, ps.y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillText(`${{i + 1}} u=${{m.u.toFixed(3)}}`, ps.x + 7, ps.y - 7);
  }}
}}

function updateMarkers() {{
  markersBody.innerHTML = markers.map((m, i) =>
    `<tr><td>${{i + 1}}</td><td>${{m.u.toFixed(4)}}</td><td>${{m.s.toFixed(3)}}</td><td>${{m.x.toFixed(3)}}</td><td>${{m.z.toFixed(3)}}</td></tr>`
  ).join('');
}}

canvas.addEventListener('mousedown', e => {{
  dragging = true;
  last = {{x: e.clientX, y: e.clientY, moved: false}};
}});

window.addEventListener('mouseup', e => {{
  if (!dragging || !last) return;
  const moved = last.moved;
  dragging = false;
  if (!moved && e.target === canvas) {{
    const rect = canvas.getBoundingClientRect();
    const w = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    const s = axisS(w.x, w.z);
    const u = axisUFromS(s);
    markers.push({{x: w.x, z: w.z, s, u}});
    markers.sort((a, b) => a.s - b.s);
    updateMarkers();
    draw();
  }}
}});

window.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  if (dragging && last) {{
    const dx = e.clientX - last.x;
    const dy = e.clientY - last.y;
    if (Math.hypot(dx, dy) > 2) last.moved = true;
    offsetX += dx;
    offsetY += dy;
    last.x = e.clientX;
    last.y = e.clientY;
    draw();
  }}
  if (e.target === canvas) {{
    const w = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    const s = axisS(w.x, w.z);
    const u = axisUFromS(s);
    hoverBox.textContent = `x: ${{w.x.toFixed(4)}}\\nz: ${{w.z.toFixed(4)}}\\ns: ${{s.toFixed(4)}}\\nu: ${{u.toFixed(4)}}`;
  }}
}});

canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouse = {{x: e.clientX - rect.left, y: e.clientY - rect.top}};
  const before = screenToWorld(mouse.x, mouse.y);
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  scale *= factor;
  const after = worldToScreen(before.x, before.z);
  offsetX += mouse.x - after.x;
  offsetY += mouse.y - after.y;
  draw();
}}, {{passive: false}});

document.getElementById('fit').addEventListener('click', fit);
document.getElementById('clear').addEventListener('click', () => {{
  markers = [];
  updateMarkers();
  draw();
}});

window.addEventListener('resize', resize);
resize();
fit();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate an interactive X-Z top view with the fish PCA axis.")
    parser.add_argument("--mesh", type=Path, default=Path("data/fish_body.mesh"))
    parser.add_argument("--out", type=Path, default=Path("data/fish_pca_topview.html"))
    args = parser.parse_args()

    points = read_medit_vertices(args.mesh)
    center, axis, s = compute_pca(points)
    html = build_html(points, center, axis, s, args.mesh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"PCA axis: {axis}")
    print(f"s range: {s.min():.6f} .. {s.max():.6f}")


if __name__ == "__main__":
    main()
