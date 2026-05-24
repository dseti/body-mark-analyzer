import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import utils
import history_manager as hm

CUSTOM_CSS = """
<style>
    [data-testid="stHeader"] {
        background-color: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stDecoration"] { display: none !important; }
    
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 999999 !important;
        background-color: #ffffff !important;
        border-radius: 4px !important;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.15) !important;
    }
    
    .block-container { padding-top: 25px !important; padding-bottom: 0px !important; }
    .main-title { font-size: 24px; font-weight: 700; color: #2c3e50; margin-bottom: 0rem; }
    .main-subtitle { font-size: 14px; font-weight: 400; color: #7f8c8d; margin-bottom: 0.5rem; }
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSlider { margin-bottom: -0.4rem; }
</style>
"""

# CRITICAL LINE: Ensure both parameters are declared here
def render_sidebar_controls(current_cfg, img=None):
    cfg = {}
    shape_options = ["None", "Circles/Dots", "Lines", "Squares", "Diamonds"]
    
    with st.sidebar:
        if img is not None:
            st.markdown("#### 🔮 Best Guess Variations")
            
            mutation_profiles = [
                {"name": "🎯 Fine Dots", "changes": {'radius_size': 15, 'threshold_val': 6, 'shape_amplify': 'Circles/Dots', 'coalesce_radius': 1, 'coalesce_intensify': 128}},
                {"name": "☁️ Diffuse Faint", "changes": {'radius_size': 121, 'threshold_val': 9, 'shape_amplify': 'None', 'coalesce_radius': 11, 'coalesce_intensify': 140}},
                {"name": "⬢ Bold Massing", "changes": {'radius_size': 75, 'threshold_val': 22, 'shape_amplify': 'Circles/Dots', 'coalesce_radius': 19, 'coalesce_intensify': 175}},
                {"name": "▬ Linear Focus", "changes": {'radius_size': 151, 'threshold_val': 12, 'shape_amplify': 'Lines', 'coalesce_radius': 1, 'coalesce_intensify': 128}}
            ]
            
            raw_h, raw_w = img.shape[:2]
            thumb_w = 140
            thumb_h = int(raw_h * (thumb_w / raw_w))
            thumb_img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            
            import image_processing as ip
            
            m_cols = st.columns(2)
            for idx, profile in enumerate(mutation_profiles):
                col_target = m_cols[idx % 2]
                with col_target:
                    mut_cfg = current_cfg.copy()
                    mut_cfg.update(profile["changes"])
                    
                    m_canvas = ip.run_abstraction_pipeline(thumb_img, mut_cfg, [])
                    st.image(m_canvas, use_container_width=True)
                    if st.button(profile["name"], key=f"apply_preset_{idx}", use_container_width=True):
                        st.session_state.cfg.update(profile["changes"])
                        hm.commit_to_history()
                        st.rerun()
            st.divider()

        with st.expander("🔬 Feature Isolation Options", expanded=True):
            cfg['radius_size'] = st.slider("Mark Extraction Radius (Size)", 3, 1001, value=current_cfg.get('radius_size', 51), step=2)
            cfg['threshold_val'] = st.slider("Extraction Threshold (Intensity)", 1, 100, value=current_cfg.get('threshold_val', 15))
        
        with st.expander("📐 Geometric Shape Filtering", expanded=True):
            cfg['shape_amplify'] = st.selectbox(
                "Target Feature to Amplify", shape_options,
                index=shape_options.index(current_cfg.get('shape_amplify', 'None'))
            )
            cfg['shape_filter_size'] = st.slider("Shape Evaluation Window Scale", 1, 31, value=current_cfg.get('shape_filter_size', 5), step=2)

        with st.expander("🔮 Object Coalescence & Massing", expanded=True):
            cfg['coalesce_radius'] = st.slider("Coalesce Bridge Width", 1, 101, value=current_cfg.get('coalesce_radius', 1), step=2)
            cfg['coalesce_intensify'] = st.slider("Coalesce Edge Intensity", 1, 254, value=current_cfg.get('coalesce_intensify', 128))

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
            st.session_state.last_click_id = None
            st.session_state.current_file = None
            st.session_state.history = []
            st.session_state.history_idx = -1
            st.rerun()
            
    return cfg

def render_header_and_history():
    h_idx = st.session_state.history_idx
    h_len = len(st.session_state.history)
    
    title_col, undo_col, redo_col = st.columns([5.5, 1.2, 1.2])
    with title_col:
        st.markdown('<div class="main-title">Body Mark Extractor</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-subtitle">A tool to produce an abstract image of body marks</div>', unsafe_allow_html=True)
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
    st.markdown("<p style='margin-top:2px; margin-bottom:2px; font-weight:600; font-size:13px; color:#2c3e50;'>🛠️ Interactive Target Guidance Toolkit:</p>", unsafe_allow_html=True)
    dock_c1, dock_c2 = st.columns([4.2, 1.8])
    with dock_c1:
        marker_type = st.radio("Active Toolkit:", ["Dot", "Skin", "Not Skin", "Paint Highlight", "Erase Highlight"], horizontal=True, label_visibility="collapsed")
    with dock_c2:
        st.session_state.brush_radius = st.slider("Brush Size", 5, 1000, st.session_state.brush_radius, label_visibility="collapsed")
    
    if st.button("🗑️ Clear All Manual Brush & Element Layers", use_container_width=True):
        st.session_state.calib_points = []
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        hm.commit_to_history()
        st.rerun()
    return marker_type

def render_input_photo_content(img, marker_type, file_name):
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Interactive Source Photo Canvas:</p>", unsafe_allow_html=True)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if np.any(st.session_state.roi_canvas == 255):
        paint_overlay = img_display.copy()
        paint_overlay[st.session_state.roi_canvas == 255] = [255, 150, 0]
        cv2.addWeighted(paint_overlay, 0.25, img_display, 0.75, 0, img_display)
        
    color_map = {"Dot": (0, 255, 255), "Skin": (0, 255, 0), "Not Skin": (255, 0, 255)}
    for p in st.session_state.calib_points:
        cv2.circle(img_display, (p["x"], p["y"]), 5, color_map.get(p["label"], (255, 255, 255)), -1)

    scaled_in_img, in_scale_factor = utils.fit_image_to_viewport(img_display, max_h=360)
    
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
            else:
                if marker_type == "Dot":
                    cv2.circle(st.session_state.roi_canvas, (cx, cy), br_raw, 255, -1)
                    
                point_removed = False
                for idx, p in enumerate(st.session_state.calib_points):
                    if np.sqrt((p['x'] - cx)**2 + (p['y'] - cy)**2) < 15:
                        st.session_state.calib_points.pop(idx)
                        point_removed = True
                        break
                if not point_removed:
                    st.session_state.calib_points.append({"x": cx, "y": cy, "label": marker_type})
            
            hm.commit_to_history() 
            st.rerun()

def render_output_photo_content(abstracted_canvas):
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Abstracted Mark Composition Canvas:</p>", unsafe_allow_html=True)
    scaled_out_img, _ = utils.fit_image_to_viewport(abstracted_canvas, max_h=360)
    st.image(scaled_out_img, use_container_width=True)