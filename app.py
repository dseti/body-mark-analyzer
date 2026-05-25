# ===================================================================
# 🛠️ CRITICAL COMPATIBILITY PATCH FOR MODERN STREAMLIT + DRAWABLE CANVAS
# ===================================================================
import streamlit.elements.image as st_image
try:
    import streamlit.elements.lib.image_utils as st_image_utils
    if not hasattr(st_image, "image_to_url") and hasattr(st_image_utils, "image_to_url"):
        st_image.image_to_url = st_image_utils.image_to_url

    try:
        from streamlit.elements.lib.layout_utils import LayoutConfig
    except ImportError:
        class LayoutConfig:
            def __init__(self, width="stretch"):
                self.width = width

    orig_image_to_url = getattr(st_image_utils, "image_to_url", None) or getattr(st_image, "image_to_url", None)
    if orig_image_to_url:
        def wrapped_image_to_url(*args, **kwargs):
            args = list(args)
            if len(args) > 1 and isinstance(args[1], int):
                args[1] = LayoutConfig(width=args[1])
            if "width" in kwargs:
                w = kwargs.pop("width")
                kwargs["layout_config"] = LayoutConfig(width=w)
            if "layout_config" in kwargs and isinstance(kwargs["layout_config"], int):
                kwargs["layout_config"] = LayoutConfig(width=kwargs["layout_config"])
            return orig_image_to_url(*args, **kwargs)
        st_image.image_to_url = wrapped_image_to_url
        if hasattr(st_image_utils, "image_to_url"):
            st_image_utils.image_to_url = wrapped_image_to_url
except Exception:
    pass
# ===================================================================

import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image
import image_processing as ip
import ui_components as ui
import history_manager as hm

st.set_page_config(page_title="Body Mark Extractor", layout="wide")
st.markdown(ui.CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_CFG = {
    'radius_size': 51, 
    'threshold_val': 15,
    'shape_amplify': 'None',
    'shape_filter_size': 5,
    'coalesce_radius': 1,
    'coalesce_intensify': 128,
    'presentation_style': "Dark Marks on Light Canvas", 
    'enable_isolation': True,
    'color_tolerance': 25
}

# Initialize global state tracking variables
for key, default in [
    ("calib_points", []), ("roi_canvas", None), ("brush_radius", 25), 
    ("current_file", None), ("cfg", DEFAULT_CFG.copy()), 
    ("history", []), ("history_idx", -1), ("shared_canvas_json", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### 🎛️ Abstraction Engine")
    uploaded_file = st.file_uploader("Upload Target Dermal File Asset", type=["jpg", "jpeg", "png"])
    st.divider()

img = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.calib_points = []
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        st.session_state.cfg = DEFAULT_CFG.copy()
        st.session_state.shared_canvas_json = None
        st.session_state.history = []
        st.session_state.history_idx = -1
        hm.commit_to_history() 
        st.rerun()

    if st.session_state.roi_canvas is None or st.session_state.roi_canvas.shape != img.shape[:2]:
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        hm.commit_to_history()

sidebar_cfg = ui.render_sidebar_controls(st.session_state.cfg, img)
if sidebar_cfg != st.session_state.cfg and sidebar_cfg:
    st.session_state.cfg.update(sidebar_cfg)

fixed_header_slot = st.container()
workspace_slot = st.container()

if uploaded_file is None:
    with workspace_slot:
        st.info("Please upload a dermal photo asset via the sidebar control panel to initialize channels.")
else:
    with fixed_header_slot:
        ui.render_header_and_history()

    with workspace_slot:
        # Create the operational three-column workspace matrix layout
        col_left, col_center, col_right = st.columns([4.2, 1.6, 4.2], gap="medium")
        
        # 1. Populate the central control console toolbar row
        with col_center:
            st.markdown("<p style='font-weight:600; font-size:13px; color:#2c3e50;'>🛠️ Studio Toolkit:</p>", unsafe_allow_html=True)
            tool_mode = st.radio(
                "Active Canvas Tool:", 
                ["Paint Highlight", "Erase Highlight", "Mark Pick", "Skin Pick", "Exclude Pick"],
                index=0, label_visibility="collapsed"
            )
            st.divider()
            st.markdown("<p style='font-weight:500; font-size:12px;'>Brush Size Selection Track</p>", unsafe_allow_html=True)
            brush_size = st.slider("Brush Dimensions", 3, 200, st.session_state.brush_radius, label_visibility="collapsed")
            st.session_state.brush_radius = brush_size

        # 2. Render the interactive Input viewport
        with col_left:
            res_left, scale_factor = ui.render_input_studio_canvas(img, tool_mode, brush_size)

        # Idempotent State Engine Sync Check for Left Canvas interactions
        if res_left.json_data and res_left.json_data != st.session_state.shared_canvas_json:
            st.session_state.shared_canvas_json = res_left.json_data
            hm.commit_to_history()
            st.rerun()

        # Parse geometric vector data structures from the current active master layer
        paint_mask, erase_mask, calib_points = ip.parse_canvas_json(st.session_state.shared_canvas_json, img.shape, scale_factor)
        st.session_state.calib_points = calib_points

        # Generate the initial base processing mask matrix preview frame
        abstract_canvas_init = ip.run_abstraction_pipeline(
            img, st.session_state.cfg, calib_points, roi_paint=paint_mask, roi_erase=erase_mask
        )

        # 3. Render the interactive Output viewport
        with col_right:
            res_right = ui.render_output_studio_canvas(abstract_canvas_init, tool_mode, brush_size)

        # Idempotent State Engine Sync Check for Right Canvas interactions
        if res_right.json_data and res_right.json_data != st.session_state.shared_canvas_json:
            st.session_state.shared_canvas_json = res_right.json_data
            hm.commit_to_history()
            st.rerun()

        # Render final combined array output pass down through presentation containers
        with col_right:
            st.divider()
            st.markdown("<p style='font-weight:500; font-size:12px; margin-bottom: 2px;'>Output Composition Layout Format Options</p>", unsafe_allow_html=True)
            
            # Repositioned format dropdown controls cleanly below output images
            presentation_style = st.selectbox(
                "Presentation Style Selection", 
                ["Dark Marks on Light Canvas", "Light Marks on Dark Canvas", "High-Visibility Overlay"],
                index=["Dark Marks on Light Canvas", "Light Marks on Dark Canvas", "High-Visibility Overlay"].index(st.session_state.cfg.get('presentation_style', "Dark Marks on Light Canvas")),
                label_visibility="collapsed", key="footer_presentation"
            )
            if presentation_style != st.session_state.cfg['presentation_style']:
                st.session_state.cfg['presentation_style'] = presentation_style
                st.rerun()
                
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            
            out_pil = Image.fromarray(abstract_canvas_init)
            buffer = io.BytesIO()
            out_pil.save(buffer, format="PNG")
            
            st.download_button(
                label="💾 Save Abstracted Canvas Image Asset",
                data=buffer.getvalue(),
                file_name=f"abstract_{st.session_state.current_file}.png",
                mime="image/png",
                use_container_width=True
            )