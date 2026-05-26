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
import time
import logging
from PIL import Image
import image_processing as ip
import ui_components as ui
import history_manager as hm

# Initialize structured backend service logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Enforce state tracking arrays early to avoid runtime race conditions
if 'img_array' not in st.session_state:
    st.session_state.img_array = None

# Whenever the upload form is active, the sidebar must remain strictly closed/collapsed
if st.session_state.img_array is None:
    st.session_state.current_sidebar_state = "collapsed"

st.set_page_config(
    page_title="Body Mark Extractor", 
    layout="wide", 
    initial_sidebar_state=st.session_state.get('current_sidebar_state', 'collapsed')
)
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

STUDIO_TOOLS = ["Select/Move", "Paint Highlight", "Erase Highlight", "Mark Pick", "Skin Pick", "Exclude Pick"]

# Initialize tracking schema structures
for key, default in [
    ("calib_points", []), ("roi_canvas", None), ("brush_radius", 25), 
    ("current_file", None), ("cfg", DEFAULT_CFG.copy()), 
    ("history", []), ("history_idx", -1), ("shared_canvas_json", {"objects": []}),
    ("canvas_version", 0), ("last_canvas_version", -1),
    ("active_tool_mode", "Select/Move"),
    ("force_open_best_guess", False),
    ("widget_triggered_rerun", False) # Reliable barrier flag against custom component race conditions
]:
    if key not in st.session_state:
        st.session_state[key] = default

def sync_tool_mode_callback():
    if "native_studio_tool_radio" in st.session_state:
        st.session_state.active_tool_mode = st.session_state.native_studio_tool_radio
        st.session_state["widget_triggered_rerun"] = True 

def sync_brush_radius_callback():
    if "studio_brush_dimension_slider" in st.session_state:
        st.session_state.brush_radius = st.session_state.studio_brush_dimension_slider
        st.session_state["widget_triggered_rerun"] = True 

def sync_presentation_style_callback():
    if "sidebar_presentation_select" in st.session_state and "cfg" in st.session_state:
        st.session_state.cfg['presentation_style'] = st.session_state.sidebar_presentation_select
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Presentation mapping flipped to '{st.session_state.sidebar_presentation_select}' safely.")

# -------------------------------------------------------------------
# CONDITION A: ONBOARDING LANDING VIEW (No Image Asset Loaded)
# -------------------------------------------------------------------
if st.session_state.img_array is None:
    with st.sidebar:
        st.markdown("<p style='color:#64748b; font-style:italic; font-size:13px; margin-top:10px;'>Upload an image asset to view output composition canvas.</p>", unsafe_allow_html=True)

    st.markdown('<div class="main-title" style="text-align: center; margin-top: 5vh;">Body Mark Extractor</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle" style="text-align: center; margin-bottom: 4vh;">Produce clean, structured high-visibility abstract maps of dermal patterns and marks.</div>', unsafe_allow_html=True)
    
    _, landing_center, _ = st.columns([1.2, 2.0, 1.2])
    with landing_center:
        st.markdown("""
        <div class="instruction-box">
            <h4 style="margin-top: 0px; color: #2c3e50;">🚀 Workflow Instructions:</h4>
            <ol style="padding-left: 20px; font-size: 14px; color: #34495e; line-height: 1.6;">
                <li><strong>Upload Photo:</strong> Drop your source file below to launch the interactive workspace.</li>
                <li><strong>Analyze & Label:</strong> Use the brush tool matrix to the right of the viewport to isolate regions or point anchors. (Double click vectors to delete).</li>
                <li><strong>Refine Settings:</strong> Adjust fine feature isolation and coalescing filters directly underneath the studio block.</li>
                <li><strong>Save Canvas Assets:</strong> Monitor real-time output maps and save assets from the <strong>Left Sidebar panel</strong>, which opens automatically on upload.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Image Asset", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            decoded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            logger.info(f"Initializing operational studio for asset: {uploaded_file.name}")
            st.session_state.img_array = decoded_img
            st.session_state.current_file = uploaded_file.name
            st.session_state.calib_points = []
            st.session_state.roi_canvas = np.zeros(decoded_img.shape[:2], dtype=np.uint8)
            st.session_state.cfg = DEFAULT_CFG.copy()
            st.session_state.shared_canvas_json = {"objects": []}
            st.session_state.history = []
            st.session_state.history_idx = -1
            st.session_state.canvas_version += 1
            st.session_state.active_tool_mode = "Select/Move"
            
            st.session_state.current_sidebar_state = "expanded" 
            st.session_state.force_open_best_guess = True       
            
            hm.commit_to_history()
            st.rerun()

# -------------------------------------------------------------------
# CONDITION B: DIGITAL STUDIO WORKSPACE (Image Cache Active)
# -------------------------------------------------------------------
else:
    img = st.session_state.img_array

    # 1. Declare Left Sidebar Content Placeholders
    with st.sidebar:
        output_image_slot = st.container()
        st.divider()
        presentation_style_slot = st.container()
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        download_button_slot = st.container()

    # 2. Split Workspace Layout Flow
    # Only load initial_drawing when canvas key changes (Undo/Redo/Upload). Keeps canvas stable during slider interactions.
    if st.session_state.canvas_version != st.session_state.last_canvas_version:
        objects = st.session_state.shared_canvas_json.get("objects", [])
        for obj in objects:
            obj["lockScalingX"] = True
            obj["lockScalingY"] = True
            obj["lockRotation"] = True
            obj["hasControls"] = False
        initial_drawing = {"objects": objects}
        st.session_state.last_canvas_version = st.session_state.canvas_version
    else:
        initial_drawing = None

    studio_layout_canvas, studio_layout_tools = st.columns([6.8, 3.2], gap="large")

    with studio_layout_canvas:
        res_left, scale_factor = ui.render_input_studio_canvas(
            img, st.session_state.active_tool_mode, st.session_state.brush_radius, 
            st.session_state.canvas_version, initial_drawing
        )

    with studio_layout_tools:
        st.markdown("<p class='section-label' style='margin-top:0px;'>🛠️ Workspace Studio Toolkit</p>", unsafe_allow_html=True)
        
        st.session_state.active_tool_mode = st.radio(
            "Studio Tool Selection Matrix",
            STUDIO_TOOLS,
            index=STUDIO_TOOLS.index(st.session_state.active_tool_mode),
            key="native_studio_tool_radio",
            on_change=sync_tool_mode_callback
        )
        
        brush_slider_disabled = "Highlight" not in st.session_state.active_tool_mode

        st.markdown("<p style='font-size: 13px; font-weight: 500; color: #475569; margin-top: 15px; margin-bottom: -5px;'>Active Stroke Radius Scale (Highlights Only)</p>", unsafe_allow_html=True)
        
        brush_size = st.slider(
            "Active Stroke Radius Scale", 3, 200, 
            st.session_state.brush_radius, 
            label_visibility="collapsed",
            disabled=brush_slider_disabled,
            key="studio_brush_dimension_slider",
            on_change=sync_brush_radius_callback
        )
        if not brush_slider_disabled:
            st.session_state.brush_radius = brush_size

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Undo/Redo Deck
        h_idx = st.session_state.history_idx
        h_len = len(st.session_state.history)
        undo_btn_col, redo_btn_col = st.columns(2)
        
        with undo_btn_col:
            if st.button("⬅️ Undo Action", disabled=(h_idx <= 0), use_container_width=True):
                hm.restore_from_history(h_idx - 1)
                st.session_state.canvas_version += 1
                st.rerun()
        with redo_btn_col:
            if st.button("➡️ Redo Action", disabled=(h_idx >= h_len - 1), use_container_width=True):
                hm.restore_from_history(h_idx + 1)
                st.session_state.canvas_version += 1
                st.rerun()

    # Sync interactive vector alterations safely
    if st.session_state.get("widget_triggered_rerun", False):
        logger.info("External widget modification active. Skipping sync cycle to protect current canvas vectors.")
    elif res_left.json_data and initial_drawing is None:
        left_objects = res_left.json_data.get("objects", [])
        shared_objects = st.session_state.shared_canvas_json.get("objects", [])
        if left_objects != shared_objects:
            logger.info("Canvas path vector modification caught.")
            for obj in left_objects:
                obj["lockScalingX"] = True
                obj["lockScalingY"] = True
                obj["lockRotation"] = True
                obj["hasControls"] = False
            st.session_state.shared_canvas_json = res_left.json_data
            hm.commit_to_history()

    # 3. Process Computer Vision Mathematical Abstraction Pipeline
    paint_mask, erase_mask, calib_points = ip.parse_canvas_json(st.session_state.shared_canvas_json, img.shape, scale_factor)
    st.session_state.calib_points = calib_points

    pipeline_start = time.time()
    abstract_canvas_init = ip.run_abstraction_pipeline(
        img, st.session_state.cfg, calib_points, roi_paint=paint_mask, roi_erase=erase_mask
    )
    logger.debug(f"Telemetry Core Engine calculation runtime execution metrics: {time.time() - pipeline_start:.4f}s")

    # 4. Populate Left Sidebar Target Containers Natively
    with output_image_slot:
        ui.render_output_studio_canvas(abstract_canvas_init)
        
    with presentation_style_slot:
        st.markdown("<p style='font-weight:500; font-size:12px; margin-bottom: 4px; color:#475569;'>Composition Presentation Style</p>", unsafe_allow_html=True)
        st.selectbox(
            "Presentation Style Selection Wrapper", 
            ["Dark Marks on Light Canvas", "Light Marks on Dark Canvas", "High-Visibility Overlay"],
            index=["Dark Marks on Light Canvas", "Light Marks on Dark Canvas", "High-Visibility Overlay"].index(st.session_state.cfg.get('presentation_style', "Dark Marks on Light Canvas")),
            label_visibility="collapsed", 
            key="sidebar_presentation_select",
            on_change=sync_presentation_style_callback 
        )
            
    with download_button_slot:
        out_pil = Image.fromarray(abstract_canvas_init)
        buffer = io.BytesIO()
        out_pil.save(buffer, format="PNG")
        
        st.download_button(
            label="💾 Save Abstracted Image Asset",
            data=buffer.getvalue(),
            file_name=f"abstract_{st.session_state.current_file}.png",
            mime="image/png",
            use_container_width=True
        )

    # 5. Render Engine Parameters Block across full width of main body stream
    sidebar_cfg = ui.render_advanced_settings_panel(st.session_state.cfg, img)
    if sidebar_cfg:
        any_sidebar_changed = any(st.session_state.cfg.get(k) != v for k, v in sidebar_cfg.items())
        if any_sidebar_changed:
            logger.info("Main column expander adjustments registered. Updating configuration state.")
            st.session_state.cfg.update(sidebar_cfg)
            hm.commit_to_history()  
            st.rerun()              

    # Clean up the barrier protection flag at the very end of the execution flow
    st.session_state["widget_triggered_rerun"] = False