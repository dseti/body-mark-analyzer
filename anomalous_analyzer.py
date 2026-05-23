import streamlit as st
import cv2
import numpy as np
import json
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

# Set up page configuration - Force wide layout optimization
st.set_page_config(page_title="Clinical Dermal Geometry Suite", layout="wide")

# Custom UI Design Overrides: Strict layout rules to eliminate vertical scrolling completely
st.markdown("""
<style>
    /* Compact main padding to fit entirely within standard monitor resolutions */
    .block-container { padding-top: 60px !important; padding-bottom: 0px !important; }
    
    /* Lock Main Header Elements */
    .main-title { font-size: 18px; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    
    /* Clean formatting for sidebar elements to keep them compressed */
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSlider {
        margin-bottom: -0.4rem;
    }
    
    /* Shrink spacing between expanders */
    .st-emotion-cache-p8by8by { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State variables
if "calib_points" not in st.session_state:
    st.session_state.calib_points = []
if "false_positives" not in st.session_state:
    st.session_state.false_positives = []
if "roi_canvas" not in st.session_state:
    st.session_state.roi_canvas = None
if "last_click_id" not in st.session_state:
    st.session_state.last_click_id = None
if "last_output_click" not in st.session_state:
    st.session_state.last_output_click = None
if "brush_radius" not in st.session_state:
    st.session_state.brush_radius = 25
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- Core Algorithmic Pipeline Functions ---

def neutralize_illumination(img, intensity_kernel):
    if intensity_kernel % 2 == 0: 
        intensity_kernel += 1
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel, cb, cr = cv2.split(ycrcb)
    illumination_map = cv2.GaussianBlur(y_channel, (intensity_kernel, intensity_kernel), 0)
    illumination_map[illumination_map == 0] = 1
    normalized_y = (y_channel.astype(np.float32) / illumination_map.astype(np.float32)) * 127.0
    normalized_y = np.clip(normalized_y, 0, 255).astype(np.uint8)
    reconstructed = cv2.merge([normalized_y, cb, cr])
    return cv2.cvtColor(reconstructed, cv2.COLOR_YCrCb2BGR)

def apply_gray_world_wb(img):
    b, g, r = cv2.split(img)
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    mean_all = (mean_b + mean_g + mean_r) / 3.0
    scale_b = mean_all / mean_b if mean_b > 0 else 1.0
    scale_g = mean_all / mean_g if mean_g > 0 else 1.0
    scale_r = mean_all / mean_r if mean_r > 0 else 1.0
    return cv2.merge([
        np.clip(b * scale_b, 0, 255).astype(np.uint8),
        np.clip(g * scale_g, 0, 255).astype(np.uint8),
        np.clip(r * scale_r, 0, 255).astype(np.uint8)
    ])

def apply_manual_levels(channel, black_point, white_point):
    if black_point >= white_point: 
        white_point = black_point + 1
    scaled = (channel.astype(np.float32) - black_point) / (white_point - black_point) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def segment_skin_roi(img, enable_isolation, custom_bounds=None):
    if not enable_isolation:
        return img, np.ones(img.shape[:2], dtype=np.uint8) * 255
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if custom_bounds is not None:
        lower_skin, upper_skin = custom_bounds
    else:
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask_smoothed = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    isolated_img = cv2.bitwise_and(img, img, mask=skin_mask_smoothed)
    return isolated_img, skin_mask_smoothed

def classify_global_layout(points):
    if len(points) < 3: 
        return "Indeterminate (Insufficient Nodes)"
    pts = np.array(points, dtype=np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx.item(), vy.item(), x0.item(), y0.item()
    
    distances_to_line = []
    for p in points: 
        distances_to_line.append(float(abs((p[0] - x0) * vy - (p[1] - y0) * vx)))
    mean_line_deviation = np.mean(distances_to_line)
    
    center = np.mean(pts, axis=0)
    distances_to_center = [float(np.linalg.norm(p - center)) for p in points]
    center_mean = np.mean(distances_to_center)
    cov_radius = np.std(distances_to_center) / center_mean if center_mean > 0 else 1
    
    if mean_line_deviation < 15.0: 
        return "Linear Vector Array (DIAL-like)"
    elif cov_radius < 0.18: 
        return "Concentric Ring Structure"
    else: 
        return "Structured Cluster Matrix (RGMP-like)"

# --- Master Processing Pipeline Engine ---
def run_diagnostic_pipeline(img, config, calib_points, false_positives, roi_canvas=None):
    stages = {}
    working_img = img.copy()
    
    if config['enable_wb']: 
        working_img = apply_gray_world_wb(working_img)
    if config['enable_flatten']: 
        working_img = neutralize_illumination(working_img, config['flatten_kernel'])
    stages['prepared'] = working_img

    custom_bounds = None
    if calib_points and config['enable_isolation']:
        skin_pts = [p for p in calib_points if p['label'] == 'Skin']
        if skin_pts:
            ycrcb_sample = cv2.cvtColor(working_img, cv2.COLOR_BGR2YCrCb)
            sampled_vals = [ycrcb_sample[min(max(0,int(p['y'])),img.shape[0]-1), min(max(0,int(p['x'])),img.shape[1]-1)] for p in skin_pts]
            sampled_vals = np.array(sampled_vals)
            lower_skin = np.clip(np.min(sampled_vals, axis=0) - 20, [0, 100, 60], [255, 255, 255]).astype(np.uint8)
            upper_skin = np.clip(np.max(sampled_vals, axis=0) + 20, [0, 255, 255], [255, 255, 255]).astype(np.uint8)
            custom_bounds = (lower_skin, upper_skin)
            
    img_isolated, skin_mask = segment_skin_roi(working_img, config['enable_isolation'], custom_bounds)
    stages['skin_mask'] = skin_mask
    
    if config['extraction_mode'] == "CIELAB a*-Channel":
        lab = cv2.cvtColor(img_isolated, cv2.COLOR_BGR2LAB)
        _, signal_channel, _ = cv2.split(lab)
    else:
        b_ch, g_ch, r_ch = cv2.split(img_isolated)
        signal_channel = cv2.subtract(r_ch, g_ch)
    
    signal_channel = apply_manual_levels(signal_channel, config['black_point'], config['white_point'])
    clahe = cv2.createCLAHE(clipLimit=config['clip_limit'], tileGridSize=(config['grid_size'], config['grid_size']))
    enhanced = clahe.apply(signal_channel)
    
    bs = config['blur_size']
    blurred = cv2.GaussianBlur(enhanced, (bs|1, bs|1), 0)
    stages['enhanced_signal'] = blurred

    fgs = config['bg_filter_size']
    bg_estimated = cv2.GaussianBlur(blurred, (fgs|1, fgs|1), 0)
    subtracted = cv2.subtract(blurred, bg_estimated)
    
    # Spatial Auto-Exposure Control Loop
    if config['enable_auto_exposure']:
        f_size = config['exposure_window_size'] | 1
        float_sub = subtracted.astype(np.float32)
        local_mean = cv2.GaussianBlur(float_sub, (f_size, f_size), 0)
        local_mean_sq = cv2.GaussianBlur(float_sub ** 2, (f_size, f_size), 0)
        local_var = np.maximum(local_mean_sq - (local_mean ** 2), 0)
        local_std = np.sqrt(local_var)
        
        mean_global_std = np.mean(local_std)
        float_sub = (float_sub / (local_std + 1.0)) * (mean_global_std + 5.0)
        subtracted = np.clip(float_sub, 0, 255).astype(np.uint8)

    stages['subtracted'] = subtracted

    active_thresh = config['threshold_val']
    if calib_points:
        dot_pts = [p for p in calib_points if p['label'] == 'Dot']
        bg_seeds = [p for p in calib_points if p['label'] in ['Skin', 'Not Skin']]
        if dot_pts and bg_seeds:
            dot_v = [subtracted[min(max(0,int(p['y'])),img.shape[0]-1), min(max(0,int(p['x'])),img.shape[1]-1)] for p in dot_pts]
            bg_v = [subtracted[min(max(0,int(p['y'])),img.shape[0]-1), min(max(0,int(p['x'])),img.shape[1]-1)] for p in bg_seeds]
            active_thresh = int((np.mean(dot_v) + np.mean(bg_v)) / 2)
    
    if false_positives:
        fp_intensities = []
        for fp in false_positives:
            fp_intensities.append(subtracted[min(max(0,int(fp['y'])),subtracted.shape[0]-1), min(max(0,int(fp['x'])),subtracted.shape[1]-1)])
        if fp_intensities:
            lowest_false_intensity = np.min(fp_intensities)
            if lowest_false_intensity <= active_thresh:
                active_thresh = max(1, int(lowest_false_intensity - 2))

    active_thresh = max(1, min(255, active_thresh))
    _, binary_mask = cv2.threshold(subtracted, active_thresh, 255, cv2.THRESH_BINARY_INV)
    if config['enable_isolation']: 
        binary_mask[skin_mask == 0] = 255
        
    if calib_points:
        not_skin_pts = [p for p in calib_points if p['label'] == 'Not Skin']
        for p in not_skin_pts:
            sig = img[min(max(0,int(p['y'])),img.shape[0]-1), min(max(0,int(p['x'])),img.shape[1]-1)]
            dist_sum = np.sum(cv2.absdiff(img, np.array(sig, dtype=np.uint8)), axis=2)
            binary_mask[dist_sum < 40] = 255 

    # Custom Painted Mask Gate
    if roi_canvas is not None and np.any(roi_canvas == 255):
        binary_mask[roi_canvas == 0] = 255
                
    # INLINE REINFORCEMENT: Force a valid blob signature at every user-taught 'Dot' coordinate
    tracking_mask = cv2.bitwise_not(binary_mask)
    if calib_points:
        for p in calib_points:
            if p['label'] == 'Dot':
                cv2.circle(tracking_mask, (int(p['x']), int(p['y'])), 3, 255, -1)
                
    contours, _ = cv2.findContours(tracking_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features, centroids = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # LEARNING OVERRIDE: Check if this contour directly houses a user-placed Dot anchor
        contains_user_dot = False
        if calib_points:
            for p in calib_points:
                if p['label'] == 'Dot':
                    if cv2.pointPolygonTest(cnt, (float(p['x']), float(p['y'])), False) >= 0:
                        contains_user_dot = True
                        break
        
        # If the user forced this seed dot, completely bypass morphological area sliders
        if contains_user_dot or (config['min_area'] < area < config['max_area']):
            perimeter = cv2.arcLength(cnt, True)
            M = cv2.moments(cnt)
            if M["m00"] != 0 and perimeter > 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                is_false_positive = False
                for fp in false_positives:
                    if np.sqrt((fp['x'] - cX)**2 + (fp['y'] - cY)**2) < 20:
                        is_false_positive = True
                        break
                if is_false_positive: 
                    continue
                
                centroids.append([cX, cY])
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                features.append({
                    "node_id": len(features),
                    "centroid_xy": [cX, cY],
                    "surface_area_pixels": float(area),
                    "border_regularity_index": float(round(circularity, 3))
                })
        
    global_layout = classify_global_layout(centroids)
    telemetry = {"active_thresh": active_thresh, "custom_skin": custom_bounds is not None}
    
    return binary_mask, stages, features, global_layout, telemetry

def fit_image_to_viewport(img, max_h=360):
    raw_h, raw_w = img.shape[:2]
    if raw_h <= max_h:
        return img, 1.0
    scale = max_h / raw_h
    new_w = int(raw_w * scale)
    scaled_img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
    return scaled_img, scale


# --- SIDEBAR CONTROLS ACCORDIONS ---

with st.sidebar:
    st.markdown("### 🛠️ Calibration Cockpit")
    uploaded_file = st.file_uploader("Upload Target Dermal File Asset", type=["jpg", "jpeg", "png"])
    
    cfg = {}
    
    with st.expander("✨ Stage A: Illumination Prep", expanded=False):
        cfg['enable_wb'] = st.checkbox("Enable White Balance", value=True)
        cfg['enable_flatten'] = st.checkbox("Enable Shading Correction", value=True)
        cfg['flatten_kernel'] = st.slider("Shadow Scan Radius", 15, 255, 101, 2)
        
    with st.expander("🧬 Stage B: Domain Skin Isolation", expanded=False):
        cfg['enable_isolation'] = st.checkbox("Isolate Active Skin ROI", value=True)
        cfg['extraction_mode'] = st.selectbox("Isolation Workspace Model", ["Red-Green Delta (R - G)", "CIELAB a*-Channel"])
        
    with st.expander("🎨 Stage C: Contrast & Filters", expanded=False):
        cfg['black_point'] = st.slider("Black Point Clip Range", 0, 254, 0)
        cfg['white_point'] = st.slider("White Point Clip Range", 1, 255, 255)
        cfg['clip_limit'] = st.slider("CLAHE Contrast Cap", 1.0, 10.0, 4.0, 0.5)
        cfg['grid_size'] = st.slider("CLAHE Matrix Block Size", 4, 64, 16, 2)
        cfg['blur_size'] = st.slider("Pore Suppression Radius", 1, 15, 5, 2)
        
    with st.expander("📏 Stage D: Spatial Auto-Exposure Limits", expanded=True):
        cfg['enable_auto_exposure'] = st.checkbox("Enable Spatial Auto-Exposure", value=True)
        cfg['exposure_window_size'] = st.slider("Auto-Exposure Block Window", 9, 151, 41, 2)
        cfg['bg_filter_size'] = st.slider("Baseline Subtraction Width", 15, 299, 149, 2)
        cfg['threshold_val'] = st.slider("Fallback Threshold Constant", 1, 255, 15)
        cfg['min_area'] = st.slider("Minimum Object Area Gate", 1, 200, 15)
        cfg['max_area'] = st.number_input("Maximum Object Area Gate", value=5000, step=100)

    st.divider()
    if st.button("🔄 Reset Global System State", use_container_width=True):
        st.session_state.calib_points = []
        st.session_state.false_positives = []
        st.session_state.roi_canvas = None
        st.session_state.last_click_id = None
        st.session_state.last_output_click = None
        st.session_state.current_file = None
        st.rerun()


# --- FIXED ABOVE-THE-FOLD INTERACTIVE STAGE ---

st.markdown('<div class="main-title">Clinical Dermal Geometry Suite — Active Tracking Workspace</div>', unsafe_allow_html=True)

col_stage1, col_stage2 = st.columns([1, 1], gap="medium")

with col_stage1:
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Field Sample Input Canvas:</p>", unsafe_allow_html=True)

with col_stage2:
    st.markdown("<p style='margin-bottom:4px; font-weight:500; font-size:13px;'>Diagnostic Segmentation Mask:</p>", unsafe_allow_html=True)

if uploaded_file is None:
    with col_stage1:
        st.info("Please upload a dermal photo asset via the sidebar tuning cockpit to initialize channels.")


if uploaded_file is not None:
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

    # --- INPUT VIEWPORT LOGIC ---
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

        # Render full-scale graphics changes directly onto the production array layers
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if np.any(st.session_state.roi_canvas == 255):
            paint_overlay = img_display.copy()
            paint_overlay[st.session_state.roi_canvas == 255] = [255, 150, 0]
            cv2.addWeighted(paint_overlay, 0.25, img_display, 0.75, 0, img_display)
            
        color_map = {"Dot": (0, 255, 255), "Skin": (0, 255, 0), "Not Skin": (255, 0, 255)}
        for p in st.session_state.calib_points:
            cv2.circle(img_display, (p["x"], p["y"]), 5, color_map.get(p["label"], (255, 255, 255)), -1)

        # COMPACT SCALING ENGINE: Compress the display frame layout to stay above-the-fold
        scaled_in_img, in_scale_factor = fit_image_to_viewport(img_display, max_h=360)
        
        # Inject dynamic graphic brush ring selectors to match the scaled display canvas metrics
        br_scaled = int(st.session_state.brush_radius * in_scale_factor)
        diam_scaled = br_scaled * 2
        
        if "Highlight" in marker_type:
            st.markdown(f"""
            <style>
                .precision-canvas img {{
                    cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='{diam_scaled}' height='{diam_scaled}' viewBox='0 0 {diam_scaled} {diam_scaled}'><circle cx='{br_scaled}' cy='{br_scaled}' r='{br_scaled-1}' stroke='%23ff9600' stroke-width='2' fill='none'/></svg>") {br_scaled} {br_scaled}, crosshair !important;
                }}
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
                .precision-canvas img { cursor: crosshair !important; }
                .output-canvas img { cursor: pointer !important; }
            </style>
            """, unsafe_allow_html=True)

        # Mount Input Canvas Component with Dynamic File-Bound Key tags
        st.markdown('<div class="precision-canvas">', unsafe_allow_html=True)
        coords_in = streamlit_image_coordinates(Image.fromarray(scaled_in_img), key=f"in_canvas_{uploaded_file.name}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if coords_in is not None:
            click_id = f"in_{coords_in['x']}_{coords_in['y']}"
            if st.session_state.last_click_id != click_id:
                st.session_state.last_click_id = click_id
                
                # RE-MAPPING VECTOR ENGINE: Map viewport coordinates back up to raw production matrix boundaries
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

    # Core Master Processing Engine Run
    final_mask, s_stage, features, layout, telemetry = run_diagnostic_pipeline(
        img, cfg, st.session_state.calib_points, st.session_state.false_positives, roi_canvas=st.session_state.roi_canvas
    )

    # --- OUTPUT VIEWPORT LOGIC ---
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

        # COMPACT SCALING ENGINE: Compress the display output layout matrix to stay above-the-fold
        scaled_out_img, out_scale_factor = fit_image_to_viewport(mask_view, max_h=360)

        st.markdown('<div class="output-canvas">', unsafe_allow_html=True)
        coords_out = streamlit_image_coordinates(Image.fromarray(scaled_out_img), key=f"out_canvas_{uploaded_file.name}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if coords_out is not None:
            click_id_out = f"out_{coords_out['x']}_{coords_out['y']}"
            if st.session_state.last_output_click != click_id_out:
                st.session_state.last_output_click = click_id_out
                
                # RE-MAPPING VECTOR ENGINE: Translate false-positive click entries safely back to original resolution
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

        # Informatics summary parameters stacked cleanly at standard footer layouts
        st.markdown("<p style='font-weight:600; font-size:14px; margin-top:2px; margin-bottom:2px;'>Real-Time Structural Informatics Summary</p>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Calculated Threshold Line", telemetry['active_thresh'])
        m2.metric("Matrix Constellation Pattern", layout)
        
        with st.expander("View Spatial Geometry Structural JSON Object"):
            st.json({"global_structural_layout": layout, "total_independent_nodes": len(features), "node_metrics_matrix": features})