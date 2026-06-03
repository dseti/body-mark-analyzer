# ===================================================================
# 🛠️ CRITICAL COMPATIBILITY PATCH FOR MODERN STREAMLIT + DRAWABLE CANVAS
# ===================================================================
import streamlit.elements.image as st_image
try:
    import streamlit.elements.lib.image_utils as st_image_utils
    
    # Safely extract reference to the existing runtime function
    current_func = getattr(st_image_utils, "image_to_url", None) or getattr(st_image, "image_to_url", None)
    
    # Direct function-level object check to fully block duplicate wrapping loops
    if current_func and not getattr(current_func, "_is_our_wrapper", False):
        try:
            from streamlit.elements.lib.layout_utils import LayoutConfig
        except ImportError:
            class LayoutConfig:
                def __init__(self, width="stretch"):
                    self.width = width

        def wrapped_image_to_url(*args, **kwargs):
            args_list = list(args)
            if len(args_list) > 1 and isinstance(args_list[1], int):
                args_list[1] = LayoutConfig(width=args_list[1])
            if "width" in kwargs:
                w = kwargs.pop("width")
                kwargs["layout_config"] = LayoutConfig(width=w)
            if "layout_config" in kwargs and isinstance(kwargs["layout_config"], int):
                kwargs["layout_config"] = LayoutConfig(width=kwargs["layout_config"])
            return current_func(*args_list, **kwargs)
        
        # Stamp the sentinel flag onto the wrapper instance itself
        wrapped_image_to_url._is_our_wrapper = True
        
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
import os
import time
import logging
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import image_processing as ip
import ui_components as ui
import history_manager as hm

# Initialize structured backend service logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Core Application Tracking Metadata Version Spec
APP_VERSION = "1.3.0"

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
    'color_tolerance': 25,
    'crop_to_mark': False,
    'crop_buffer': 20,
    'rotation': 0,
    'grayscale_steps': 2
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
    ("widget_triggered_rerun", False) 
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Define Modal Window Content for README documentation
def render_readme_content():
    try:
        readme_path = os.path.join(os.path.dirname(__file__), "README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content)
    except Exception as e:
        st.error(f"Could not load README.md documentation: {e}")

if hasattr(st, "dialog"):
    @st.dialog("Documentation & Research Context", width="large")
    def show_readme_modal():
        render_readme_content()
else:
    @st.experimental_dialog("Documentation & Research Context", width="large")
    def show_readme_modal():
        render_readme_content()

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

def sync_crop_to_mark_callback():
    if "sidebar_crop_to_mark_check" in st.session_state and "cfg" in st.session_state:
        st.session_state.cfg['crop_to_mark'] = st.session_state.sidebar_crop_to_mark_check
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Crop output to mark toggled to {st.session_state.sidebar_crop_to_mark_check}")

# Bidirectional Sidebar Synchronization Callbacks
def sync_sidebar_crop_num_to_slide():
    if "num_sidebar_crop_buffer" in st.session_state and "cfg" in st.session_state:
        val = st.session_state.num_sidebar_crop_buffer
        st.session_state.slide_sidebar_crop_buffer = val
        st.session_state.cfg['crop_buffer'] = val
        hm.commit_to_history()
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Crop buffer adjusted via text input to {val}px")

def sync_sidebar_crop_slide_to_num():
    if "slide_sidebar_crop_buffer" in st.session_state and "cfg" in st.session_state:
        val = st.session_state.slide_sidebar_crop_buffer
        st.session_state.num_sidebar_crop_buffer = val
        st.session_state.cfg['crop_buffer'] = val
        hm.commit_to_history()
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Crop buffer adjusted via slider to {val}px")

def sync_sidebar_rotation_num_to_slide():
    if "num_sidebar_rotation" in st.session_state and "cfg" in st.session_state:
        val = st.session_state.num_sidebar_rotation
        st.session_state.slide_sidebar_rotation = val
        st.session_state.cfg['rotation'] = val
        hm.commit_to_history()
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Composition rotation set via text input to {val}°")

def sync_sidebar_rotation_slide_to_num():
    if "slide_sidebar_rotation" in st.session_state and "cfg" in st.session_state:
        val = st.session_state.slide_sidebar_rotation
        st.session_state.num_sidebar_rotation = val
        st.session_state.cfg['rotation'] = val
        hm.commit_to_history()
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Composition rotation set via slider to {val}°")

def sync_sidebar_steps_num_to_slide():
    if "num_sidebar_grayscale_steps" in st.session_state and "cfg" in st.session_state:
        val = st.session_state.num_sidebar_grayscale_steps
        st.session_state.slide_sidebar_grayscale_steps = val
        st.session_state.cfg['grayscale_steps'] = val
        hm.commit_to_history()
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Grayscale steps set via text input to {val}")

def sync_sidebar_steps_slide_to_num():
    if "slide_sidebar_grayscale_steps" in st.session_state and "cfg" in st.session_state:
        val = st.session_state.slide_sidebar_grayscale_steps
        st.session_state.num_sidebar_grayscale_steps = val
        st.session_state.cfg['grayscale_steps'] = val
        hm.commit_to_history()
        st.session_state["widget_triggered_rerun"] = True
        logger.info(f"Grayscale steps set via slider to {val}")

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

        # 📍 Active Canvas Elements & Pins Management Panel
        st.markdown("<p class='section-label'>📍 Active Canvas Elements & Pins</p>", unsafe_allow_html=True)
        canvas_objects = st.session_state.shared_canvas_json.get("objects", [])
        
        if not canvas_objects:
            st.caption("No elements or pins placed yet.")
        else:
            with st.container(height=200):
                for idx, obj in enumerate(canvas_objects):
                    obj_type = obj.get("type")
                    stroke = obj.get("stroke", "")
                    
                    if obj_type == "circle":
                        label = "Mark Pick"
                        icon = "🟡"
                        if "0, 255, 0" in stroke:
                            label = "Skin Pick"
                            icon = "🟢"
                        elif "255, 165, 0" in stroke:
                            label = "Exclude Pick"
                            icon = "🟠"
                        
                        radius = obj.get("radius", 4)
                        left = int(obj.get("left", 0) + radius)
                        top = int(obj.get("top", 0) + radius)
                        display_text = f"{icon} **{label}** ({left}, {top})"
                    elif obj_type == "path":
                        label = "Paint Highlight" if "0, 255, 255" in stroke else "Erase Highlight"
                        icon = "🖌️" if "0, 255, 255" in stroke else "🧽"
                        display_text = f"{icon} **{label}**"
                    else:
                        display_text = f"📦 **{obj_type.capitalize()}**"
                        
                    c_el1, c_el2 = st.columns([3.5, 1.5])
                    with c_el1:
                        st.markdown(f"<span style='font-size:12px;'>{display_text}</span>", unsafe_allow_html=True)
                    with c_el2:
                        if st.button("🗑️", key=f"del_el_{idx}", use_container_width=True):
                            st.session_state.shared_canvas_json["objects"].pop(idx)
                            st.session_state.canvas_version += 1
                            hm.commit_to_history()
                            st.session_state["widget_triggered_rerun"] = True
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

    # 4. Inject Metadata Payload & Assemble Structural PNG Byte Packages
    out_pil = Image.fromarray(abstract_canvas_init)
    
    png_metadata = PngInfo()
    png_metadata.add_text("AppName", "Body Mark Analyzer Studio")
    png_metadata.add_text("AppVersion", APP_VERSION)
    png_metadata.add_text("AppConfig", json.dumps(st.session_state.cfg))
    png_metadata.add_text("GenerationTimestamp", str(time.time()))
    
    buffer = io.BytesIO()
    out_pil.save(buffer, format="PNG", pnginfo=png_metadata)
    png_bytes = buffer.getvalue()

    # 5. Populate Left Sidebar Target Containers Natively
    with output_image_slot:
        ui.render_output_studio_canvas(png_bytes)
        
    with presentation_style_slot:
        st.markdown("<p style='font-weight:500; font-size:12px; margin-bottom: 4px; color:#475569;'>Composition Presentation Style</p>", unsafe_allow_html=True)
        styles_list = ["Dark Marks on Light Canvas", "Light Marks on Dark Canvas", "Black Marks on Transparent Canvas", "High-Visibility Overlay"]
        st.selectbox(
            "Presentation Style Selection Wrapper", 
            styles_list,
            index=styles_list.index(st.session_state.cfg.get('presentation_style', "Dark Marks on Light Canvas")),
            label_visibility="collapsed", 
            key="sidebar_presentation_select",
            on_change=sync_presentation_style_callback 
        )
        
        st.checkbox(
            "Crop output to mark",
            value=st.session_state.cfg.get('crop_to_mark', False),
            key="sidebar_crop_to_mark_check",
            on_change=sync_crop_to_mark_callback
        )
        
        if st.session_state.cfg.get('crop_to_mark', False):
            st.markdown("<p style='font-weight:500; font-size:12px; margin-top: 8px; margin-bottom: 4px; color:#475569;'>Crop Margin Buffer (px)</p>", unsafe_allow_html=True)
            
            # Authoritative state synchronization barrier for history/presets
            current_crop_buffer = st.session_state.cfg.get('crop_buffer', 20)
            if st.session_state.get('num_sidebar_crop_buffer') != current_crop_buffer or st.session_state.get('slide_sidebar_crop_buffer') != current_crop_buffer:
                st.session_state.num_sidebar_crop_buffer = current_crop_buffer
                st.session_state.slide_sidebar_crop_buffer = current_crop_buffer
            
            col_m1, col_m2 = st.columns([1.7, 1.3])
            with col_m2:
                st.number_input(
                    "Crop Margin Buffer Num input", 0, 200,
                    step=5,
                    label_visibility="collapsed",
                    key="num_sidebar_crop_buffer",
                    on_change=sync_sidebar_crop_num_to_slide
                )
            with col_m1:
                st.slider(
                    "Crop Margin Buffer Scope slider", 0, 200,
                    step=5,
                    label_visibility="collapsed",
                    key="slide_sidebar_crop_buffer",
                    on_change=sync_sidebar_crop_slide_to_num
                )
            
        st.markdown("<p style='font-weight:500; font-size:12px; margin-top: 8px; margin-bottom: 4px; color:#475569;'>Final Output Rotation (Degrees)</p>", unsafe_allow_html=True)
        
        # Authoritative state synchronization barrier for rotation history/resets
        current_rotation = st.session_state.cfg.get('rotation', 0)
        if st.session_state.get('num_sidebar_rotation') != current_rotation or st.session_state.get('slide_sidebar_rotation') != current_rotation:
            st.session_state.num_sidebar_rotation = current_rotation
            st.session_state.slide_sidebar_rotation = current_rotation

        col_r1, col_r2 = st.columns([1.7, 1.3])
        with col_r2:
            st.number_input(
                "Final Output Rotation Num input", 0, 360,
                step=1,
                label_visibility="collapsed",
                key="num_sidebar_rotation",
                on_change=sync_sidebar_rotation_num_to_slide
            )
        with col_r1:
            st.slider(
                "Final Output Rotation Angle slider", 0, 360,
                step=1,
                label_visibility="collapsed",
                key="slide_sidebar_rotation",
                on_change=sync_sidebar_rotation_slide_to_num
            )

        st.markdown("<p style='font-weight:500; font-size:12px; margin-top: 8px; margin-bottom: 4px; color:#475569;'>Grayscale Output Steps (2 = Binary)</p>", unsafe_allow_html=True)
        
        # Authoritative state synchronization barrier for grayscale steps
        current_steps = st.session_state.cfg.get('grayscale_steps', 2)
        if st.session_state.get('num_sidebar_grayscale_steps') != current_steps or st.session_state.get('slide_sidebar_grayscale_steps') != current_steps:
            st.session_state.num_sidebar_grayscale_steps = current_steps
            st.session_state.slide_sidebar_grayscale_steps = current_steps

        col_s1, col_s2 = st.columns([1.7, 1.3])
        with col_s2:
            st.number_input(
                "Grayscale Steps Num input", 2, 10,
                step=1,
                label_visibility="collapsed",
                key="num_sidebar_grayscale_steps",
                on_change=sync_sidebar_steps_num_to_slide
            )
        with col_s1:
            st.slider(
                "Grayscale Steps slider", 2, 10,
                step=1,
                label_visibility="collapsed",
                key="slide_sidebar_grayscale_steps",
                on_change=sync_sidebar_steps_slide_to_num
            )
            
    with download_button_slot:
        base_name, _ = os.path.splitext(st.session_state.current_file)
        st.download_button(
            label="💾 Save Abstracted Image Asset",
            data=png_bytes,
            file_name=f"abstract_{base_name}.png",
            mime="image/png",
            use_container_width=True
        )
        
        # 📖 "Learn More" Trigger button rendering right below the Save button
        if st.button("Learn More", use_container_width=True):
            show_readme_modal()

    # 6. Render Engine Parameters Block across full width of main body stream
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
