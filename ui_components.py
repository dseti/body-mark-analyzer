import streamlit as st
import cv2
import numpy as np
from PIL import Image
import utils
import history_manager as hm

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please execute environment configuration dependency sync: `pip install streamlit-drawable-canvas`")
    st.stop()

# Reverted completely to your original stable CSS overlay setup
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

def precise_slider(label, min_v, max_v, default_v, step=1):
    """Combines native sliders with a typeable number field side-by-side"""
    c1, c2 = st.columns([3, 1])
    with c2:
        num_val = st.number_input(label, min_value=min_v, max_value=max_v, value=default_v, step=step, label_visibility="collapsed", key=f"num_{label}")
    with c1:
        slide_val = st.slider(label, min_value=min_v, max_value=max_v, value=num_val, step=step, label_visibility="collapsed", key=f"slide_{label}")
    return slide_val

def render_sidebar_controls(current_cfg, img=None):
    cfg = {}
    shape_options = ["None", "Circles/Dots", "Lines", "Squares", "Diamonds"]
    
    with st.sidebar:
        if img is not None:
            with st.expander("🔮 Best Guess Variations", expanded=False):
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
                        st.image(m_canvas, width="stretch")
                        if st.button(profile["name"], key=f"apply_preset_{idx}", width="stretch"):
                            st.session_state.cfg.update(profile["changes"])
                            hm.commit_to_history()
                            st.rerun()

        with st.expander("🔬 Feature Isolation Options", expanded=True):
            st.caption("Mark Extraction Radius (Size)")
            cfg['radius_size'] = precise_slider("radius_size", 3, 1001, current_cfg.get('radius_size', 51), 2)
            st.caption("Extraction Threshold (Intensity)")
            cfg['threshold_val'] = precise_slider("threshold_val", 1, 100, current_cfg.get('threshold_val', 15), 1)
            
        with st.expander("🪄 Color Isolation Gating (Magic Wand)", expanded=True):
            cfg['enable_isolation'] = st.checkbox("Enable Skin ROI Isolation", value=current_cfg.get('enable_isolation', True))
            st.caption("Color Selection Node Tolerance")
            cfg['color_tolerance'] = precise_slider("color_tolerance", 1, 100, current_cfg.get('color_tolerance', 25), 1)
        
        with st.expander("📐 Geometric Shape Filtering", expanded=True):
            cfg['shape_amplify'] = st.selectbox(
                "Target Feature to Amplify", shape_options,
                index=shape_options.index(current_cfg.get('shape_amplify', 'None'))
            )
            st.caption("Shape Evaluation Window Scale")
            cfg['shape_filter_size'] = precise_slider("shape_filter_size", 1, 31, current_cfg.get('shape_filter_size', 5), 2)

        with st.expander("🔮 Object Coalescence & Massing", expanded=True):
            st.caption("Coalesce Bridge Width")
            cfg['coalesce_radius'] = precise_slider("coalesce_radius", 1, 101, current_cfg.get('coalesce_radius', 1), 2)
            st.caption("Coalesce Edge Intensity")
            cfg['coalesce_intensify'] = precise_slider("coalesce_intensify", 1, 254, current_cfg.get('coalesce_intensify', 128), 1)

        st.divider()
        if st.button("🔄 Reset Application State", width="stretch"):
            st.session_state.calib_points = []
            st.session_state.roi_canvas = None
            st.session_state.shared_canvas_json = {"objects": []}
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
        if st.button("⬅️ Undo State", disabled=(h_idx <= 0), width="stretch"):
            hm.restore_from_history(h_idx - 1)
            st.rerun()
    with redo_col:
        if st.button("➡️ Redo State", disabled=(h_idx >= h_len - 1), width="stretch"):
            hm.restore_from_history(h_idx + 1)
            st.rerun()
    st.divider()

def get_tool_stroke_settings(tool_mode, brush_size):
    """Calculates active mouse modes and drawing configurations for the canvas engine"""
    drawing_mode = "freedraw" if "Highlight" in tool_mode else "point"
    point_radius = 4
    
    stroke_color = "rgba(0, 255, 255, 0.4)"       # Transparent Cyan Paint Highlight
    if tool_mode == "Erase Highlight": 
        stroke_color = "rgba(255, 0, 255, 0.4)"   # Transparent Magenta Erase Highlight Ink
    elif tool_mode == "Mark Pick": 
        stroke_color = "rgba(255, 255, 0, 1.0)"   # Solid Yellow Vector Point Dot
    elif tool_mode == "Skin Pick": 
        stroke_color = "rgba(0, 255, 0, 1.0)"     # Solid Green Skin Point Dot
    elif tool_mode == "Exclude Pick": 
        stroke_color = "rgba(255, 165, 0, 1.0)"   # Solid Orange Exclude Point Dot

    active_width = brush_size if drawing_mode == "freedraw" else point_radius
    return drawing_mode, stroke_color, active_width

def render_input_studio_canvas(img, tool_mode, brush_size):
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Interactive Source Photo Canvas:</p>", unsafe_allow_html=True)
    
    drawing_mode, stroke_color, active_width = get_tool_stroke_settings(tool_mode, brush_size)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scaled_img, scale_factor = utils.fit_image_to_viewport(img_display, max_h=380)
    pil_image = Image.fromarray(scaled_img)

    # Sanitize initial drawing to isolate objects and prevent background metadata collisions
    clean_drawing = None
    if st.session_state.shared_canvas_json and "objects" in st.session_state.shared_canvas_json:
        clean_drawing = {"objects": st.session_state.shared_canvas_json["objects"]}

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.0)",
        stroke_width=active_width,
        stroke_color=stroke_color,
        background_image=pil_image,
        update_streamlit=True,
        height=scaled_img.shape[0],
        width=scaled_img.shape[1],
        drawing_mode=drawing_mode,
        initial_drawing=clean_drawing,
        key=f"canvas_left_{st.session_state.current_file}",
    )
    return canvas_result, scale_factor

def render_output_studio_canvas(abstracted_canvas, tool_mode, brush_size):
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Abstracted Mark Composition Canvas (Read-Only Matrix):</p>", unsafe_allow_html=True)
    
    _, stroke_color, active_width = get_tool_stroke_settings(tool_mode, brush_size)
    scaled_img, scale_factor = utils.fit_image_to_viewport(abstracted_canvas, max_h=380)
    pil_image = Image.fromarray(scaled_img)

    # Sanitize initial drawing to isolate vector objects onto output composition
    clean_drawing = None
    if st.session_state.shared_canvas_json and "objects" in st.session_state.shared_canvas_json:
        clean_drawing = {"objects": st.session_state.shared_canvas_json["objects"]}

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.0)",
        stroke_width=active_width,
        stroke_color=stroke_color,
        background_image=pil_image,
        update_streamlit=False, # CRITICAL ARCHITECTURAL FIX: Completely decouple from Streamlit rerun cycle
        height=scaled_img.shape[0],
        width=scaled_img.shape[1],
        drawing_mode="transform", # Strictly non-writable panning/zooming observation mode
        initial_drawing=clean_drawing,
        key=f"canvas_right_{st.session_state.current_file}",
    )
    return canvas_result