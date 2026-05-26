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

# Clean, production-ready workspace styles with all hacky radio CSS overrides removed
CUSTOM_CSS = """
<style>
    [data-testid="stHeader"] {
        background-color: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stDecoration"] { display: none !important; }
    
    /* Clean close viewport padding rules */
    .block-container { padding-top: 20px !important; padding-bottom: 0px !important; }
    
    /* Typography Overrides */
    .main-title { font-size: 32px; font-weight: 800; color: #1e293b; letter-spacing: -0.5px; }
    .main-subtitle { font-size: 15px; font-weight: 400; color: #64748b; }
    .section-label { font-size: 14px; font-weight: 700; color: #334155; margin-top: 20px; margin-bottom: 5px; }
    
    /* Document Onboarding Instruction Box */
    .instruction-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 25px;
    }
    
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSlider { margin-bottom: -0.4rem; }
</style>
"""

def precise_slider(label, min_v, max_v, default_v, step=1, key_prefix=""):
    """Combines native sliders with a typeable number field side-by-side with bidirectional sync"""
    unique_key = f"{key_prefix}_{label}"
    num_key = f"num_{unique_key}"
    slide_key = f"slide_{unique_key}"
    last_auth_key = f"last_auth_{unique_key}"

    # Handle external authoritative updates (e.g., presets click or history undo/redo)
    if last_auth_key not in st.session_state or st.session_state[last_auth_key] != default_v:
        st.session_state[num_key] = default_v
        st.session_state[slide_key] = default_v
        st.session_state[last_auth_key] = default_v

    # State synchronization logic
    def sync_num_to_slide():
        st.session_state[slide_key] = st.session_state[num_key]

    def sync_slide_to_num():
        st.session_state[num_key] = st.session_state[slide_key]

    c1, c2 = st.columns([3, 1])
    with c2:
        num_val = st.number_input(
            label, 
            min_value=min_v, 
            max_value=max_v, 
            key=num_key, 
            step=step, 
            label_visibility="collapsed",
            on_change=sync_num_to_slide
        )
    with c1:
        slide_val = st.slider(
            label, 
            min_value=min_v, 
            max_value=max_v, 
            key=slide_key, 
            step=step, 
            label_visibility="collapsed",
            on_change=sync_slide_to_num
        )
        
    st.session_state[last_auth_key] = slide_val
    return slide_val

def render_advanced_settings_panel(current_cfg, img=None):
    """Renders abstraction engine configurations natively within main studio stream layout. All expanders default closed."""
    cfg = {}
    
    st.markdown("<p class='section-label'>⚙️ Advanced Abstraction Engine Control Settings</p>", unsafe_allow_html=True)
    
    # Process transactional single-shot automatic opening for Best Guess Variations
    best_guess_expanded = False
    if st.session_state.get('force_open_best_guess', False):
        best_guess_expanded = True
        st.session_state.force_open_best_guess = False 

    if img is not None:
        with st.expander("🔮 Best Guess Variations", expanded=best_guess_expanded):
            mutation_profiles = [
                {"name": "🎯 Fine Dots", "changes": {'radius_size': 15, 'threshold_val': 6, 'coalesce_radius': 1, 'coalesce_intensify': 128}},
                {"name": "☁️ Diffuse Faint", "changes": {'radius_size': 121, 'threshold_val': 9, 'coalesce_radius': 11, 'coalesce_intensify': 140}},
                {"name": "⬢ Bold Massing", "changes": {'radius_size': 75, 'threshold_val': 22, 'coalesce_radius': 19, 'coalesce_intensify': 175}},
                {"name": "▬ Linear Focus", "changes": {'radius_size': 151, 'threshold_val': 12, 'coalesce_radius': 1, 'coalesce_intensify': 128}}
            ]
            
            raw_h, raw_w = img.shape[:2]
            thumb_w = 140
            thumb_h = int(raw_h * (thumb_w / raw_w))
            thumb_img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            
            import image_processing as ip
            
            m_cols = st.columns(4)
            for idx, profile in enumerate(mutation_profiles):
                col_target = m_cols[idx % 4]
                with col_target:
                    mut_cfg = current_cfg.copy()
                    mut_cfg.update(profile["changes"])
                    m_canvas = ip.run_abstraction_pipeline(thumb_img, mut_cfg, [])
                    st.image(m_canvas, use_container_width=True)
                    if st.button(profile["name"], key=f"apply_preset_{idx}", use_container_width=True):
                        st.session_state.cfg.update(profile["changes"])
                        hm.commit_to_history()
                        st.rerun()

    with st.expander("🔬 Feature Isolation Options", expanded=False):
        st.caption("Mark Extraction Radius (Size)")
        cfg['radius_size'] = precise_slider("radius_size", 3, 1001, current_cfg.get('radius_size', 51), 2, key_prefix="studio_isol")
        st.caption("Extraction Threshold (Intensity)")
        cfg['threshold_val'] = precise_slider("threshold_val", 1, 100, current_cfg.get('threshold_val', 15), 1, key_prefix="studio_isol")
        
    with st.expander("🪄 Color Isolation Gating (Magic Wand)", expanded=False):
        cfg['enable_isolation'] = st.checkbox("Enable Skin ROI Isolation", value=current_cfg.get('enable_isolation', True))
        st.caption("Color Selection Node Tolerance")
        cfg['color_tolerance'] = precise_slider("color_tolerance", 1, 100, current_cfg.get('color_tolerance', 25), 1, key_prefix="studio_color")

    with st.expander("🔮 Object Coalescence & Massing", expanded=False):
        st.caption("Coalesce Bridge Width")
        cfg['coalesce_radius'] = precise_slider("coalesce_radius", 1, 101, current_cfg.get('coalesce_radius', 1), 2, key_prefix="studio_coal")
        st.caption("Coalesce Edge Intensity")
        cfg['coalesce_intensify'] = precise_slider("coalesce_intensify", 1, 254, current_cfg.get('coalesce_intensify', 128), 1, key_prefix="studio_coal")

    st.divider()
    
    if st.session_state.get("confirm_reset_state", False):
        st.warning("⚠️ Are you sure you want to completely reset the application state?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, Confirm Reset", use_container_width=True, key="btn_confirm_reset_true"):
                st.session_state.confirm_reset_state = False
                st.session_state.calib_points = []
                st.session_state.roi_canvas = None
                st.session_state.shared_canvas_json = {"objects": []}
                st.session_state.history = []
                st.session_state.history_idx = -1
                st.session_state.canvas_version += 1
                st.session_state.current_file = None
                st.session_state.img_array = None
                st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True, key="btn_confirm_reset_false"):
                st.session_state.confirm_reset_state = False
                st.rerun()
    else:
        if st.button("🔄 Reset Application State", use_container_width=True):
            st.session_state.confirm_reset_state = True
            st.rerun()
        
    return cfg

def get_tool_stroke_settings(tool_mode, brush_size):
    """Calculates active mouse modes and drawing configurations for the canvas engine."""
    if tool_mode == "Select/Move":
        drawing_mode = "transform"
    elif "Highlight" in tool_mode:
        drawing_mode = "freedraw"
    else:
        drawing_mode = "point"

    point_radius = 4
    stroke_color = "rgba(0, 255, 255, 0.4)" 
    
    if tool_mode == "Erase Highlight": 
        stroke_color = "rgba(255, 0, 255, 0.4)" 
    elif tool_mode == "Mark Pick": 
        stroke_color = "rgba(255, 255, 0, 1.0)" 
    elif tool_mode == "Skin Pick": 
        stroke_color = "rgba(0, 255, 0, 1.0)" 
    elif tool_mode == "Exclude Pick": 
        stroke_color = "rgba(255, 165, 0, 1.0)" 
    elif tool_mode == "Select/Move":
        stroke_color = "rgba(0, 0, 0, 0.0)" 

    active_width = brush_size if drawing_mode == "freedraw" else point_radius
    return drawing_mode, stroke_color, active_width

def render_input_studio_canvas(img, tool_mode, brush_size, canvas_version, initial_drawing):
    st.markdown("<p class='section-label' style='margin-top:0px;'>Interactive Source Photo Canvas Viewport:</p>", unsafe_allow_html=True)
    
    drawing_mode, stroke_color, active_width = get_tool_stroke_settings(tool_mode, brush_size)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scaled_img, scale_factor = utils.fit_image_to_viewport(img_display, max_h=440)
    pil_image = Image.fromarray(scaled_img)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.0)",
        stroke_width=active_width,
        stroke_color=stroke_color,
        background_image=pil_image,
        update_streamlit=True,
        height=scaled_img.shape[0],
        width=scaled_img.shape[1],
        drawing_mode=drawing_mode,
        initial_drawing=initial_drawing,
        display_toolbar=False,
        key=f"canvas_left_{st.session_state.current_file}_{canvas_version}",
    )
    
    st.markdown("<p style='font-size:11px; color:#64748b; margin-top: -6px; margin-bottom:10px;'><i>Tip: In Select/Move mode, double-click an item to delete it.</i></p>", unsafe_allow_html=True)
    
    return canvas_result, scale_factor

def render_output_studio_canvas(abstracted_canvas):
    st.markdown("<p style='margin-bottom:6px; font-weight:700; font-size:14px; color:#1e293b;'>Abstracted Composition Output</p>", unsafe_allow_html=True)
    st.image(abstracted_canvas, use_container_width=True)
