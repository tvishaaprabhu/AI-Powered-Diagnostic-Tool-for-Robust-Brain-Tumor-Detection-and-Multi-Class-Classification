import streamlit as st
import streamlit.components.v1 as components
import base64
import json
from PIL import Image
import io
import numpy as np

def draw_bbox(img_array: np.ndarray, key: str = "bbox") -> list:
    """
    Renders an interactive canvas for drawing a bounding box.
    Returns [x_min, y_min, x_max, y_max] in original image coordinates,
    or None if no box drawn yet.
    """
    # Convert to RGB PIL and encode as base64
    if len(img_array.shape) == 2:
        pil_img = Image.fromarray(img_array).convert("RGB")
    else:
        pil_img = Image.fromarray(img_array)

    # Resize for display
    DISPLAY_W = 512
    orig_w, orig_h = pil_img.size
    display_h = int(orig_h * DISPLAY_W / orig_w)
    display_img = pil_img.resize((DISPLAY_W, display_h))

    buf = io.BytesIO()
    display_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    scale_x = orig_w / DISPLAY_W
    scale_y = orig_h / display_h

    html = f"""
    <div>
        <p style="font-family:sans-serif;font-size:13px;color:#555;">
            Click and drag on the image to draw a bounding box around the tumor.
        </p>
        <canvas id="canvas_{key}" width="{DISPLAY_W}" height="{display_h}"
            style="border:2px solid #00bfff;cursor:crosshair;display:block;"></canvas>
        <div id="coords_{key}" style="font-family:monospace;font-size:12px;
            color:#333;margin-top:6px;">No box drawn yet.</div>
        <button onclick="confirmBox_{key}()" style="margin-top:8px;padding:8px 20px;
            background:#00bfff;color:white;border:none;border-radius:4px;
            cursor:pointer;font-size:14px;">Confirm Bounding Box</button>
    </div>

    <script>
    (function() {{
        const canvas = document.getElementById('canvas_{key}');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        let startX, startY, isDrawing = false;
        let box = null;

        img.onload = function() {{ ctx.drawImage(img, 0, 0); }};
        img.src = 'data:image/png;base64,{img_b64}';

        canvas.addEventListener('mousedown', e => {{
            const r = canvas.getBoundingClientRect();
            startX = e.clientX - r.left;
            startY = e.clientY - r.top;
            isDrawing = true;
        }});

        canvas.addEventListener('mousemove', e => {{
            if (!isDrawing) return;
            const r = canvas.getBoundingClientRect();
            const x = e.clientX - r.left;
            const y = e.clientY - r.top;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            ctx.strokeStyle = '#00ffff';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 3]);
            ctx.strokeRect(startX, startY, x - startX, y - startY);
            box = {{
                x1: Math.round(Math.min(startX, x)),
                y1: Math.round(Math.min(startY, y)),
                x2: Math.round(Math.max(startX, x)),
                y2: Math.round(Math.max(startY, y))
            }};
            document.getElementById('coords_{key}').innerText =
                `Box: (${{box.x1}}, ${{box.y1}}) → (${{box.x2}}, ${{box.y2}})`;
        }});

        canvas.addEventListener('mouseup', () => {{ isDrawing = false; }});

        window.confirmBox_{key} = function() {{
            if (!box) {{ alert('Please draw a bounding box first.'); return; }}
            // Scale back to original image coords
            const scaled = {{
                x1: Math.round(box.x1 * {scale_x}),
                y1: Math.round(box.y1 * {scale_y}),
                x2: Math.round(box.x2 * {scale_x}),
                y2: Math.round(box.y2 * {scale_y})
            }};
            // Send to Streamlit via query param workaround
            const url = new URL(window.location.href);
            url.searchParams.set('bbox_{key}', JSON.stringify(scaled));
            window.history.replaceState(null, '', url.toString());
            document.getElementById('coords_{key}').innerText =
                `✅ Confirmed: (${{scaled.x1}}, ${{scaled.y1}}) → (${{scaled.x2}}, ${{scaled.y2}})`;
        }};
    }})();
    </script>
    """

    components.html(html, height=display_h + 100)

    # Read bbox from query params if set
    params = st.query_params
    param_key = f"bbox_{key}"
    if param_key in params:
        try:
            coords = json.loads(params[param_key])
            return [coords["x1"], coords["y1"], coords["x2"], coords["y2"]]
        except Exception:
            return None
    return None
