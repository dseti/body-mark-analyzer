import cv2
import numpy as np

def run_abstraction_pipeline(img, config, calib_points, roi_canvas=None, exposure_canvas=None):
    working_img = img.copy()
    h, w = working_img.shape[:2]
    
    # 1. Base Domain Skin Masking
    skin_mask = np.ones((h, w), dtype=np.uint8) * 255
    if config['enable_isolation']:
        ycrcb = cv2.cvtColor(working_img, cv2.COLOR_BGR2YCrCb)
        skin_pts = [p for p in calib_points if p['label'] == 'Skin']
        
        if skin_pts: 
            sampled_vals = [ycrcb[min(max(0, int(p['y'])), h-1), min(max(0, int(p['x'])), w-1)] for p in skin_pts]
            sampled_vals = np.array(sampled_vals)
            lower_bounds = np.clip(np.min(sampled_vals, axis=0) - 25, [0, 100, 60], [255, 255, 255]).astype(np.uint8)
            upper_bounds = np.clip(np.max(sampled_vals, axis=0) + 25, [0, 255, 255], [255, 255, 255]).astype(np.uint8)
            skin_mask = cv2.inRange(ycrcb, lower_bounds, upper_bounds)
        else: 
            skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # 2. Extract Targeted Contrast Channel (Green)
    _, signal_channel, _ = cv2.split(working_img)
    
    # Non-destructive manual Dodge & Burn local adjustments layer integration
    if exposure_canvas is not None:
        signal_channel = np.clip(signal_channel.astype(np.int16) + exposure_canvas, 0, 255).astype(np.uint8)
    
    # 3. High-Pass Spatial Frequency Subtraction
    r_val = config['radius_size'] | 1
    bg_map = cv2.GaussianBlur(signal_channel, (r_val, r_val), 0)
    diff = cv2.subtract(bg_map, signal_channel)
    
    # 4. Abstraction Intensity Threshold
    _, binary_mask = cv2.threshold(diff, config['threshold_val'], 255, cv2.THRESH_BINARY)
    
    # 5. Integrate Live Anchor Overrides & Highlight Gates
    if config['enable_isolation']:
        binary_mask[skin_mask == 0] = 0
        
    not_skin_pts = [p for p in calib_points if p['label'] == 'Not Skin']
    for p in not_skin_pts:
        sig_color = working_img[min(max(0, int(p['y'])), h-1), min(max(0, int(p['x'])), w-1)]
        dist_field = np.sum(cv2.absdiff(working_img, np.array(sig_color, dtype=np.uint8)), axis=2)
        binary_mask[dist_field < 45] = 0 

    if roi_canvas is not None and np.any(roi_canvas == 255):
        binary_mask[roi_canvas == 0] = 0

    # 6. Apply Canvas Presentation Output Styles
    if config['presentation_style'] == "Dark Marks on Light Canvas":
        abstract_img = cv2.bitwise_not(binary_mask)
        output_canvas = cv2.cvtColor(abstract_img, cv2.COLOR_GRAY2RGB)
    elif config['presentation_style'] == "Light Marks on Dark Canvas":
        output_canvas = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    else: 
        if exposure_canvas is not None:
            for i in range(3):
                working_img[:,:,i] = np.clip(working_img[:,:,i].astype(np.int16) + exposure_canvas, 0, 255).astype(np.uint8)
        output_canvas = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
        output_canvas[binary_mask == 255] = [255, 110, 0]
        
    return output_canvas