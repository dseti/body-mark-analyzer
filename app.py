import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# Import local decoupled modules
import image_processing as ip
import ui_components as ui
import utils

# Set up page configuration optimization
st.set_page_config(page_title="Clinical Dermal Geometry Suite", layout="wide")
st.markdown(ui.CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State variables
for key, default in [
    ("calib_points", []), ("false_positives", []), ("roi_canvas", None),
    ("last_click_id", None), ("last_output_click", None), 
    ("brush_radius", 25), ("current_file", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Render Sidebar Interface
uploaded_file, cfg = ui.render_sidebar_controls()

st.markdown('<div class="main-title">Clinical Dermal Geometry Suite — Active Tracking Workspace</div>', unsafe_allow_html=True)
col_stage1, col_stage2 = st.columns([1, 1], gap="medium")

with col_stage1:
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Field Sample Input Canvas:</p>", unsafe_allow_html=True)
with col_stage2:
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Diagnostic Segmentation Mask:</p>", unsafe_allow_html=True)

if uploaded_file is None:
    with col_stage1:
        st.info("Please upload a dermal photo asset via the sidebar tuning cockpit to initialize channels.")
else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Auto-Reset state when a user replaces the current file asset
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.calib_points = []
        st.session_state.false_positives = []
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
        st.session_state.last_click_id = None
        st.session_state.last_output_click = None
        st.rerun()

    if st.session_state.roi_canvas is None or st.session_state.roi_canvas.shape != img.shape[:2]:
        st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)

    # --- INPUT VIEWPORT LAYER ---
    with col_stage1:
        dock_c1, dock_c2, dock_c3 = st.columns([2.5, 1.5, 1])
        with dock_c1:
            marker_type = st.radio("Active Toolkit:", ["Dot", "Skin", "Not Skin", "Paint Highlight", "Erase Highlight"], horizontal=True, label_visibility="collapsed")
        with dock_c2:
            st.session_state.brush_radius = st.slider("Brush Radius Size", 5, 100, st.session_state.brush_radius, label_visibility="collapsed")
        with dock_c3:
            if st.button("🗑️ Clear Mask", use_container_width=True):
                st.session_state.calib_points = []
                st.session_state.roi_canvas = np.zeros(img.shape[:2], dtype=np.uint8)
                st.rerun()

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
        coords_in = streamlit_image_coordinates(Image.fromarray(scaled_in_img), key=f"in_canvas_{uploaded_file.name}")
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
                st.rerun()

    # Master Execution Engine processing
    final_mask, s_stage, features, layout, telemetry = ip.run_diagnostic_pipeline(
        img, cfg, st.session_state.calib_points, st.session_state.false_positives, roi_canvas=st.session_state.roi_canvas
    )

    # --- OUTPUT VIEWPORT LAYER ---
    with col_stage2:
        mask_view = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
        oc1, oc2 = st.columns([2, 1])
        with oc1:
            show_ann = st.checkbox("Display Overlay Layers", value=True)
        with oc2:
            st.markdown(f"**Extracted Features:** {len(features)}")
        
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
        coords_out = streamlit_image_coordinates(Image.fromarray(scaled_out_img), key=f"out_canvas_{uploaded_file.name}")
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
                st.rerun()

        st.markdown("<p style='font-weight:600; font-size:14px; margin-top:2px; margin-bottom:2px;'>Real-Time Structural Informatics Summary</p>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Calculated Threshold Line", telemetry['active_thresh'])
        m2.metric("Matrix Constellation Pattern", layout)
        
        with st.expander("View Spatial Geometry Structural JSON Object"):
            st.json({"global_structural_layout": layout, "total_independent_nodes": len(features), "node_metrics_matrix": features})