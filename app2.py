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
    'enable_isolation': True
}

for key, default in [
    ("calib_points", []), ("roi_canvas", None), ("last_click_id", None), 
    ("brush_radius", 25), ("current_file", None), ("cfg", DEFAULT_CFG.copy()), 
    ("history", []), ("history_idx", -1)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 1. Mount persistent Upload form block onto sidebar container layer
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
        st.session_state.last_click_id = None
        st.session_state.history = []
        st.session_state.history_idx = -1
        hm.commit_to_history() 
        st.rerun()

    if st.session_state.roi_canvas is None or st.session_state.roi_canvas.shape != img.shape[:2]:
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        hm.commit_to_history()

# 2. Render remaining configurations and variation thumbnails bound to image instance
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
        col_left, col_right = st.columns([1, 1], gap="medium")
        input_photo_container = col_left.container()
        selectors_container = col_left.container()
        output_photo_container = col_right.container()
        save_button_container = col_right.container()

    with selectors_container:
        marker_type = ui.render_selectors_content(img)

    abstract_canvas = ip.run_abstraction_pipeline(
        img, st.session_state.cfg, st.session_state.calib_points, roi_canvas=st.session_state.roi_canvas
    )

    with input_photo_container:
        ui.render_input_photo_content(img, marker_type, uploaded_file.name)

    with output_photo_container:
        ui.render_output_photo_content(abstract_canvas)

    with save_button_container:
        out_pil = Image.fromarray(abstract_canvas)
        buffer = io.BytesIO()
        out_pil.save(buffer, format="PNG")
        
        st.download_button(
            label="💾 Save Abstracted Canvas Image Asset",
            data=buffer.getvalue(),
            file_name=f"abstract_{st.session_state.current_file}.png",
            mime="image/png",
            use_container_width=True
        )