import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import utils
import image_processing as ip
import history_manager as hm

CUSTOM_CSS = """
<style>
    /* Completely hide the Streamlit top header bar and decoration accent line */
    [data-testid="stHeader"], header {
        visibility: hidden;
        display: none;
        height: 0px;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }

    /* Compress the core view container padding so your layout hits the very top */
    .block-container { 
        padding-top: 15px !important; 
        padding-bottom: 0px !important; 
    }
    
    .main-title { font-size: 20px; font-weight: 700; color: #1f77b4; margin-bottom: 0rem; }
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSlider {
        margin-bottom: -0.4rem;
    }
    .st-emotion-cache-p8by8by { margin-bottom: 0.25rem; }
</style>
"""

def render_sidebar_controls(current_cfg):
    cfg = {}
    with st.sidebar:
        st.markdown("### 🛠️ Calibration Cockpit")
        uploaded_file = st.file_uploader("Upload Target Dermal File Asset", type=["jpg", "jpeg", "png"])
        
        with st.expander("✨ Stage A: Illumination Prep", expanded=False):
            cfg['enable_wb'] = st.checkbox("Enable White Balance", value=current_cfg.get('enable_wb', True))
            cfg['enable_flatten'] = st.checkbox("Enable Shading Correction", value=current_cfg.get('enable_flatten', True))
            cfg['flatten_kernel'] = st.slider("Shadow Scan Radius", 15, 255, value=current_cfg.get('flatten_kernel', 101), step=2)
            
        with st.expander("🧬 Stage B: Domain Skin Isolation", expanded=False):
            cfg['enable_isolation'] = st.checkbox("Isolate Active Skin ROI", value=current_cfg.get('enable_isolation', True))
            cfg['extraction_mode'] = st.selectbox(
                "Isolation Workspace Model", 
                ["Red-Green Delta (R - G)", "CIELAB a*-Channel"],
                index=0 if current_cfg.get('extraction_mode', "Red-Green Delta (R - G)") == "Red-Green Delta (R - G)" else 1
            )
            
        with st.expander("🎨 Stage C: Contrast & Filters", expanded=False):
            cfg['black_point'] = st.slider("Black Point Clip Range", 0, 254, value=current_cfg.get('black_point', 0))
            cfg['white_point'] = st.slider("White Point Clip Range", 1, 255, value=current_cfg.get('white_point', 255))
            cfg['clip_limit'] = st.slider("CLAHE Contrast Cap", 1.0, 10.0, value=current_cfg.get('clip_limit', 4.0), step=0.5)
            cfg['grid_size'] = st.slider("CLAHE Matrix Block Size", 4, 64, value=current_cfg.get('grid_size', 16), step=2)
            cfg['blur_size'] = st.slider("Pore Suppression Radius", 1, 15, value=current_cfg.get('blur_size', 5), step=2)
            
        with st.expander("📏 Stage D: Spatial Auto-Exposure Limits", expanded=True):
            cfg['enable_auto_exposure'] = st.checkbox("Enable Spatial Auto-Exposure", value=current_cfg.get('enable_auto_exposure', True))
            cfg['exposure_window_size'] = st.slider("Auto-Exposure Block Window", 9, 151, value=current_cfg.get('exposure_window_size', 41), step=2)
            cfg['bg_filter_size'] = st.slider("Baseline Subtraction Width", 15, 299, value=current_cfg.get('bg_filter_size', 149), step=2)
            cfg['threshold_val'] = st.slider("Fallback Threshold Constant", 1, 255, value=current_cfg.get('threshold_val', 15))
            cfg['min_area'] = st.slider("Minimum Object Area Gate", 1, 200, value=current_cfg.get('min_area', 15))
            cfg['max_area'] = st.number_input("Maximum Object Area Gate", value=current_cfg.get('max_area', 5000), step=100)

        st.divider()
        if st.button("🔄 Reset Global System State", use_container_width=True):
            st.session_state.calib_points = []
            st.session_state.false_positives = []
            st.session_state.roi_canvas = None
            st.session_state.last_click_id = None
            st.session_state.last_output_click = None
            st.session_state.current_file = None
            st.session_state.history = []
            st.session_state.history_idx = -1
            st.rerun()
            
    return uploaded_file, cfg

def render_header_and_history():
    """Renders the top fixed horizontal dashboard bar integrating Title & Undo/Redo/Save controllers."""
    h_idx = st.session_state.history_idx
    h_len = len(st.session_state.history)
    
    title_col, undo_col, redo_col, save_col = st.columns([4.5, 1.2, 1.2, 1.5])
    with title_col:
        st.markdown('<div class="main-title">Clinical Dermal Geometry Workspace</div>', unsafe_allow_html=True)
    with undo_col:
        if st.button("⬅️ Undo Action", disabled=(h_idx <= 0), use_container_width=True, key="hdr_undo"):
            hm.restore_from_history(h_idx - 1)
            st.rerun()
    with redo_col:
        if st.button("➡️ Redo Action", disabled=(h_idx >= h_len - 1), use_container_width=True, key="hdr_redo"):
            hm.restore_from_history(h_idx + 1)
            st.rerun()
    with save_col:
        if st.button("💾 Save State", use_container_width=True, key="hdr_save"):
            hm.commit_to_history()
            st.success("Saved")
    st.divider()

def render_selectors_content(img):
    """Renders the workspace input toolkit options directly below the Input Canvas."""
    st.markdown("<p style='margin-top:8px; margin-bottom:2px; font-weight:600; font-size:13px; color:#1f77b4;'>🛠️ Input Target Selection Toolkit:</p>", unsafe_allow_html=True)
    dock_c1, dock_c2 = st.columns([3, 2])
    with dock_c1:
        marker_type = st.radio("Active Toolkit:", ["Dot", "Skin", "Not Skin", "Paint Highlight", "Erase Highlight"], horizontal=True, label_visibility="collapsed")
    with dock_c2:
        st.session_state.brush_radius = st.slider("Brush Radius Size", 5, 100, st.session_state.brush_radius, label_visibility="collapsed")
    
    if st.button("🗑️ Clear Live Highlights", use_container_width=True):
        st.session_state.calib_points = []
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        hm.commit_to_history()
        st.rerun()
    return marker_type

def render_output_controls_content(features):
    """Renders diagnostic metrics adjustments directly below the Segmentation Canvas."""
    st.markdown("<p style='margin-top:8px; margin-bottom:2px; font-weight:600; font-size:13px; color:#1f77b4;'>📊 Diagnostics Display Parameters:</p>", unsafe_allow_html=True)
    oc1, oc2 = st.columns([1.5, 1])
    with oc1:
        show_ann = st.checkbox("Display Overlay Layers", value=True)
    with oc2:
        st.markdown(f"**Extracted Features:** `{len(features)}`")
    return show_ann

def render_input_photo_content(img, marker_type, file_name):
    """Renders the Input Photo view layer and hooks resolution scale mapping clicks."""
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Field Sample Input Canvas:</p>", unsafe_allow_html=True)
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.any(st.session_state.roi_canvas == 255):
        paint_overlay = img_display.copy()
        paint_overlay[st.session_state.roi_canvas == 255] = [255, 150, 0]
        cv2.addWeighted(paint_overlay, 0.25, img_display, 0.75, 0, img_display)
        
    color_map = {"Dot": (0, 255, 255), "Skin": (0, 255, 0), "Not Skin": (255, 0, 255)}
    for p in st.session_state.calib_points:
        cv2.circle(img_display, (p["x"], p["y"]), 5, color_map.get(p["label"], (255, 255, 255)), -1)

    scaled_in_img, in_scale_factor = utils.fit_image_to_viewport(img_display, max_h=360)
    br_scaled = int(st.session_state.brush_radius * in_scale_factor)
    diam_scaled = br_scaled * 2
    
    if "Highlight" in marker_type:
        st.markdown(f"<style>.precision-canvas img {{ cursor: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='{diam_scaled}' height='{diam_scaled}' viewBox='0 0 {diam_scaled} {diam_scaled}'><circle cx='{br_scaled}' cy='{br_scaled}' r='{br_scaled-1}' stroke='%23ff9600' stroke-width='2' fill='none'/></svg>\") {br_scaled} {br_scaled}, crosshair !important; }}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>.precision-canvas img { cursor: crosshair !important; } .output-canvas img { cursor: pointer !important; }</style>", unsafe_allow_html=True)

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

def render_output_photo_content(final_mask, features, show_ann, file_name):
    """Renders the Segmented Output Photo view layer and tracks anomaly adjustments."""
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Diagnostic Segmentation Mask:</p>", unsafe_allow_html=True)
    mask_view = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
    
    if show_ann:
        for f in features: 
            cv2.drawMarker(mask_view, (int(f['centroid_xy'][0]), int(f['centroid_xy'][1])), (0, 165, 255), cv2.MARKER_CROSS, 12, 2)
        for fp in st.session_state.false_positives:
            cv2.circle(mask_view, (fp["x"], fp["y"]), 5, (255, 0, 0), -1)
            cv2.circle(mask_view, (fp["x"], fp["y"]), 8, (0, 0, 255), 1)
        if np.any(st.session_state.roi_canvas == 255):
            contours_roi, _ = cv2.findContours(st.session_state.roi_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_view, contours_roi, -1, (255, 150, 0), 1)

    scaled_out_img, out_scale_factor = utils.fit_image_to_viewport(mask_view, max_h=360)

    st.markdown('<div class="output-canvas">', unsafe_allow_html=True)
    coords_out = streamlit_image_coordinates(Image.fromarray(scaled_out_img), key=f"out_canvas_{file_name}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if coords_out is not None:
        click_id_out = f"out_{coords_out['x']}_{coords_out['y']}"
        if st.session_state.last_output_click != click_id_out:
            st.session_state.last_output_click = click_id_out
            cx_o = int(coords_out['x'] / out_scale_factor)
            cy_o = int(coords_out['y'] / out_scale_factor)
            
            fp_removed = False
            for idx, fp in enumerate(st.session_state.false_positives):
                if np.sqrt((fp['x'] - cx_o)**2 + (fp['y'] - cy_o)**2) < 15:
                    st.session_state.false_positives.pop(idx)
                    fp_removed = True
                    break
            if not fp_removed:
                st.session_state.false_positives.append({"x": cx_o, "y": cy_o})
            
            hm.commit_to_history() 
            st.rerun()

def render_mutation_matrix(img):
    """Renders background simulated configurations dynamically mapped to the current state of annotations."""
    st.markdown("<p style='font-weight:600; font-size:14px; margin-bottom:8px;'>🔮 Calibration Mutation Options (Click preset button to apply)</p>", unsafe_allow_html=True)
    
    mutation_profiles = [
        {"changes": {'threshold_val': 8, 'clip_limit': 6.0, 'min_area': 5, 'bg_filter_size': 99}},
        {"changes": {'extraction_mode': "CIELAB a*-Channel", 'clip_limit': 5.0, 'threshold_val': 12, 'blur_size': 5}},
        {"changes": {'blur_size': 9, 'threshold_val': 22, 'min_area': 25, 'bg_filter_size': 199}},
        {"changes": {'min_area': 80, 'bg_filter_size': 251, 'threshold_val': 18}}
    ]

    raw_h, raw_w = img.shape[:2]
    thumb_w, thumb_h = 180, 135
    scale_x = thumb_w / raw_w
    scale_y = thumb_h / raw_h

    # Re-map full-resolution annotations down to thumbnail-resolution vectors
    scaled_calib = []
    for p in st.session_state.calib_points:
        scaled_calib.append({
            "x": int(p["x"] * scale_x),
            "y": int(p["y"] * scale_y),
            "label": p["label"]
        })

    scaled_fps = []
    for fp in st.session_state.false_positives:
        scaled_fps.append({
            "x": int(fp["x"] * scale_x),
            "y": int(fp["y"] * scale_y)
        })

    if st.session_state.roi_canvas is not None:
        scaled_roi = cv2.resize(st.session_state.roi_canvas, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
    else:
        scaled_roi = None

    mut_cols = st.columns(4)
    thumb_img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    
    for i, profile in enumerate(mutation_profiles):
        with mut_cols[i]:
            mut_cfg = st.session_state.cfg.copy()
            mut_cfg.update(profile["changes"])
            
            # Run background simulations bound to the current annotation state matrix
            m_mask, _, _, _, _ = ip.run_diagnostic_pipeline(
                thumb_img, mut_cfg, scaled_calib, scaled_fps, roi_canvas=scaled_roi
            )
            
            # Minimal layout: Just the computed photo option and button
            st.image(m_mask, use_container_width=True)
            if st.button(f"Apply Preset {i+1}", key=f"apply_mut_{i}", use_container_width=True):
                st.session_state.cfg.update(profile["changes"])
                hm.commit_to_history()
                st.rerun()

def render_informatics(telemetry, layout, features):
    """Displays real-time geometric and system performance metrics."""
    st.divider()
    st.markdown("<p style='font-weight:600; font-size:14px; margin-top:2px; margin-bottom:2px;'>Real-Time Structural Informatics Summary</p>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Calculated Threshold Line", telemetry['active_thresh'])
    m2.metric("Matrix Constellation Pattern", layout)
    
    with st.expander("View Spatial Geometry Structural JSON Object"):
        st.json({"global_structural_layout": layout, "total_independent_nodes": len(features), "node_metrics_matrix": features})