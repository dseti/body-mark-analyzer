import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import utils
import history_manager as hm

CUSTOM_CSS = """
<style>
    [data-testid="stHeader"], header { visibility: hidden; display: none; height: 0px; }
    div[data-testid="stDecoration"] { display: none; }
    .block-container { padding-top: 15px !important; padding-bottom: 0px !important; }
    .main-title { font-size: 20px; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem; }
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSlider { margin-bottom: -0.4rem; }
</style>
"""

def render_sidebar_controls(current_cfg):
    cfg = {}
    with st.sidebar:
        st.markdown("### 🎛️ Abstraction Engine")
        uploaded_file = st.file_uploader("Upload Target Dermal File Asset", type=["jpg", "jpeg", "png"])
        st.divider()
        
        cfg['radius_size'] = st.slider("Mark Extraction Radius (Size)", 3, 151, value=current_cfg.get('radius_size', 51), step=2)
        cfg['threshold_val'] = st.slider("Extraction Threshold (Intensity)", 1, 100, value=current_cfg.get('threshold_val', 15))
        
        st.divider()
        cfg['presentation_style'] = st.selectbox(
            "Visual Canvas Presentation", 
            ["Dark Marks on Light Canvas", "Light Marks on Dark Canvas", "High-Visibility Overlay"],
            index=["Dark Marks on Light Canvas", "Light Marks on Dark Canvas", "High-Visibility Overlay"].index(current_cfg.get('presentation_style', "Dark Marks on Light Canvas"))
        )
        cfg['enable_isolation'] = st.checkbox("Enable Skin ROI Isolation", value=current_cfg.get('enable_isolation', True))
        
        st.divider()
        if st.button("🔄 Reset Global Application State", use_container_width=True):
            st.session_state.calib_points = []
            st.session_state.roi_canvas = None
            st.session_state.exposure_canvas = None
            st.session_state.last_click_id = None
            st.session_state.current_file = None
            st.session_state.history = []
            st.session_state.history_idx = -1
            st.rerun()
            
    return uploaded_file, cfg

def render_header_and_history():
    h_idx = st.session_state.history_idx
    h_len = len(st.session_state.history)
    
    title_col, undo_col, redo_col = st.columns([5.5, 1.2, 1.2])
    with title_col:
        st.markdown('<div class="main-title">🎨 Dermal Feature Abstraction Studio</div>', unsafe_allow_html=True)
    with undo_col:
        if st.button("⬅️ Undo State", disabled=(h_idx <= 0), use_container_width=True):
            hm.restore_from_history(h_idx - 1)
            st.rerun()
    with redo_col:
        if st.button("➡️ Redo State", disabled=(h_idx >= h_len - 1), use_container_width=True):
            hm.restore_from_history(h_idx + 1)
            st.rerun()
    st.divider()

def render_selectors_content(img):
    st.markdown("<p style='margin-top:2px; margin-bottom:2px; font-weight:600; font-size:13px; color:#2c3e50;'>🛠️ Interactive Guidance & Exposure Editing Toolkit:</p>", unsafe_allow_html=True)
    dock_c1, dock_c2 = st.columns([4.2, 1.8])
    with dock_c1:
        marker_type = st.radio("Active Toolkit:", ["Dot", "Skin", "Not Skin", "Paint Highlight", "Erase Highlight", "Dodge", "Burn"], horizontal=True, label_visibility="collapsed")
    with dock_c2:
        st.session_state.brush_radius = st.slider("Brush Size", 5, 100, st.session_state.brush_radius, label_visibility="collapsed")
    
    if st.button("🗑️ Clear All Manual Brush & Element Layers", use_container_width=True):
        st.session_state.calib_points = []
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        st.session_state.exposure_canvas = np.zeros(img.shape[:2], dtype=np.int16)
        hm.commit_to_history()
        st.rerun()
    return marker_type

def render_input_photo_content(img, marker_type, file_name):
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Interactive Source Photo Canvas:</p>", unsafe_allow_html=True)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Overlay manual exposure alterations (Dodge / Burn modifications) onto original display layer
    if st.session_state.exposure_canvas is not None:
        for i in range(3):
            img_display[:,:,i] = np.clip(img_display[:,:,i].astype(np.int16) + st.session_state.exposure_canvas, 0, 255).astype(np.uint8)
    
    if np.any(st.session_state.roi_canvas == 255):
        paint_overlay = img_display.copy()
        paint_overlay[st.session_state.roi_canvas == 255] = [255, 150, 0]
        cv2.addWeighted(paint_overlay, 0.25, img_display, 0.75, 0, img_display)
        
    color_map = {"Dot": (0, 255, 255), "Skin": (0, 255, 0), "Not Skin": (255, 0, 255)}
    for p in st.session_state.calib_points:
        cv2.circle(img_display, (p["x"], p["y"]), 5, color_map.get(p["label"], (255, 255, 255)), -1)

    scaled_in_img, in_scale_factor = utils.fit_image_to_viewport(img_display, max_h=360)
    
    # Dynamic Cursor UI Engine - Outputs a scaling circle matching the active tool layout dimensions
    br_scaled = max(1, int(st.session_state.brush_radius * in_scale_factor))
    diam_scaled = br_scaled * 2
    
    st.markdown(f"<style>.precision-canvas img {{ cursor: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='{diam_scaled}' height='{diam_scaled}' viewBox='0 0 {diam_scaled} {diam_scaled}'><circle cx='{br_scaled}' cy='{br_scaled}' r='{br_scaled-1}' stroke='%23ff9600' stroke-width='2' fill='none'/></svg>\") {br_scaled} {br_scaled}, crosshair !important; }}</style>", unsafe_allow_html=True)

    st.markdown('<div class="precision-canvas">', unsafe_allow_html=True)
    coords_in = streamlit_image_coordinates(Image.fromarray(scaled_in_img), key=f"in_canvas_{file_name}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if coords_in is not None:
        click_id = f"in_{coords_in['x']}_{coords_in['y']}"
        if st.session_state.last_click_id != click_id:
            st.session_state.last_click_id = click_id
            cx = int(coords_in['x'] / in_scale_factor)
            cy = int(coords_in['y'] / in_scale_factor)
            br_raw = st.session_state.brush_radius
            
            if marker_type == "Paint Highlight":
                cv2.circle(st.session_state.roi_canvas, (cx, cy), br_raw, 255, -1)
            elif marker_type == "Erase Highlight":
                cv2.circle(st.session_state.roi_canvas, (cx, cy), br_raw, 0, -1)
            elif marker_type == "Dodge":
                tmp_mask = np.zeros(st.session_state.exposure_canvas.shape, dtype=np.uint8)
                cv2.circle(tmp_mask, (cx, cy), br_raw, 255, -1)
                st.session_state.exposure_canvas[tmp_mask == 255] = np.clip(st.session_state.exposure_canvas[tmp_mask == 255] + 20, -255, 255)
            elif marker_type == "Burn":
                tmp_mask = np.zeros(st.session_state.exposure_canvas.shape, dtype=np.uint8)
                cv2.circle(tmp_mask, (cx, cy), br_raw, 255, -1)
                st.session_state.exposure_canvas[tmp_mask == 255] = np.clip(st.session_state.exposure_canvas[tmp_mask == 255] - 20, -255, 255)
            else:
                if marker_type == "Dot":
                    cv2.circle(st.session_state.roi_canvas, (cx, cy), br_raw, 255, -1)
                    
                point_removed = False
                for idx, p in enumerate(st.session_state.calib_points):
                    if np.sqrt((p['x'] - cx)**2 + (p['y'] - cy)**2) < 15:
                        st.session_state.calib_points.pop(idx)
                        point_removed = True
                        break
                if not_point_removed := not point_removed:
                    st.session_state.calib_points.append({"x": cx, "y": cy, "label": marker_type})
            
            hm.commit_to_history() 
            st.rerun()

def render_output_photo_content(abstracted_canvas):
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Abstracted Mark Composition Canvas:</p>", unsafe_allow_html=True)
    scaled_out_img, _ = utils.fit_image_to_viewport(abstracted_canvas, max_h=360)
    st.image(scaled_out_img, use_container_width=False)