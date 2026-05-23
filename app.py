import streamlit as st
import cv2
import numpy as np

# Import clean decoupled sub-modules
import image_processing as ip
import ui_components as ui
import history_manager as hm

# Configure structural layout options
st.set_page_config(page_title="Clinical Dermal Geometry Suite", layout="wide")
st.markdown(ui.CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_CFG = {
    'enable_wb': True, 'enable_flatten': True, 'flatten_kernel': 101,
    'enable_isolation': True, 'extraction_mode': "Red-Green Delta (R - G)",
    'black_point': 0, 'white_point': 255, 'clip_limit': 4.0, 'grid_size': 16, 'blur_size': 5,
    'enable_auto_exposure': True, 'exposure_window_size': 41, 'bg_filter_size': 149,
    'threshold_val': 15, 'min_area': 15, 'max_area': 5000
}

# Initialize state structures dynamically
for key, default in [
    ("calib_points", []), ("false_positives", []), ("roi_canvas", None),
    ("last_click_id", None), ("last_output_click", None), 
    ("brush_radius", 25), ("current_file", None),
    ("cfg", DEFAULT_CFG.copy()), ("history", []), ("history_idx", -1)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Render sidebar parameters mapped straight into the active configuration
uploaded_file, sidebar_cfg = ui.render_sidebar_controls(st.session_state.cfg)
if sidebar_cfg != st.session_state.cfg:
    st.session_state.cfg.update(sidebar_cfg)

# ==========================================
# 🏛️ DECLARE PAGE VISUAL HIERARCHY CONTAINERS
# ==========================================
fixed_header_slot = st.container()
workspace_slot = st.container()
mutations_slot = st.container()
informatics_slot = st.container()

if uploaded_file is None:
    with workspace_slot:
        st.info("Please upload a dermal photo asset via the sidebar tuning cockpit to initialize channels.")
else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Initialize file session parameters and establish the initial history index
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.calib_points = []
        st.session_state.false_positives = []
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        st.session_state.cfg = DEFAULT_CFG.copy()
        st.session_state.last_click_id = None
        st.session_state.last_output_click = None
        st.session_state.history = []
        st.session_state.history_idx = -1
        hm.commit_to_history() 
        st.rerun()

    if st.session_state.roi_canvas is None or st.session_state.roi_canvas.shape != img.shape[:2]:
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        hm.commit_to_history()

    # 1. Mount integrated Fixed Header row
    with fixed_header_slot:
        ui.render_header_and_history()

    # 2. Divide core workspace layout into 50/50 vertical streams
    with workspace_slot:
        col_left, col_right = st.columns([1, 1], gap="medium")
        
        # Segment internal slots to align display parameters beneath photos
        input_photo_container = col_left.container()
        selectors_container = col_left.container()
        
        output_photo_container = col_right.container()
        controls_container = col_right.container()

    # 3. Process left parameters first to extract selection tool choices ahead of canvas processing
    with selectors_container:
        marker_type = ui.render_selectors_content(img)

    # 4. Evaluate production diagnostics using current session properties
    final_mask, s_stage, features, layout, telemetry = ip.run_diagnostic_pipeline(
        img, st.session_state.cfg, st.session_state.calib_points, st.session_state.false_positives, roi_canvas=st.session_state.roi_canvas
    )

    # 5. Populate right display settings below the output canvas
    with controls_container:
        show_ann = ui.render_output_controls_content(features)

    # 6. Push high-fidelity canvases to top rows within their respective layout columns
    with input_photo_container:
        ui.render_input_photo_content(img, marker_type, uploaded_file.name)

    with output_photo_container:
        ui.render_output_photo_content(final_mask, features, show_ann, uploaded_file.name)

    # 7. Mount the 4 Mutations Panel row below primary photo blocks
    with mutations_slot:
        st.divider()
        ui.render_mutation_matrix(img)

    # 8. Render technical analytics data at bottom footer
    with informatics_slot:
        ui.render_informatics(telemetry, layout, features)