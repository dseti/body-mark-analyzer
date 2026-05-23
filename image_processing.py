import cv2
import numpy as np

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

    if roi_canvas is not None and np.any(roi_canvas == 255):
        binary_mask[roi_canvas == 0] = 255
                
    tracking_mask = cv2.bitwise_not(binary_mask)
    if calib_points:
        for p in calib_points:
            if p['label'] == 'Dot':
                cv2.circle(tracking_mask, (int(p['x']), int(p['y'])), 3, 255, -1)
                
    contours, _ = cv2.findContours(tracking_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features, centroids = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        contains_user_dot = False
        if calib_points:
            for p in calib_points:
                if p['label'] == 'Dot':
                    if cv2.pointPolygonTest(cnt, (float(p['x']), float(p['y'])), False) >= 0:
                        contains_user_dot = True
                        break
        
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