import cv2
import numpy as np

def get_shape_kernel(shape_type, size):
    if size % 2 == 0: 
        size += 1
    if shape_type == "Circles/Dots":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape_type == "Squares":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape_type == "Lines":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    elif shape_type == "Diamonds":
        kernel = np.zeros((size, size), dtype=np.uint8)
        radius = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - radius) + abs(j - radius) <= radius:
                    kernel[i, j] = 1
        return kernel
    return None

def run_abstraction_pipeline(img, config, calib_points, roi_canvas=None):
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
    
    # Pre-Processor Dynamic Contrast Stretch
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    signal_channel = clahe.apply(signal_channel)
    
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

    # 6. Geometric Shape Amplification
    f_size = config.get('shape_filter_size', 5) | 1
    if config.get('shape_amplify', 'None') != 'None':
        a_kernel = get_shape_kernel(config['shape_amplify'], f_size)
        if a_kernel is not None:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, a_kernel)
            binary_mask = cv2.dilate(binary_mask, a_kernel, iterations=1)

    # 7. Coalescence Blur & Intensification (Alpha Clip)
    if config.get('coalesce_radius', 1) > 1:
        c_blur = config['coalesce_radius'] | 1
        blurred_structure = cv2.GaussianBlur(binary_mask, (c_blur, c_blur), 0)
        _, binary_mask = cv2.threshold(blurred_structure, 255 - config['coalesce_intensify'], 255, cv2.THRESH_BINARY)

    # 8. Apply Canvas Presentation Output Styles
    if config['presentation_style'] == "Dark Marks on Light Canvas":
        abstract_img = cv2.bitwise_not(binary_mask)
        output_canvas = cv2.cvtColor(abstract_img, cv2.COLOR_GRAY2RGB)
    elif config['presentation_style'] == "Light Marks on Dark Canvas":
        output_canvas = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    else: 
        output_canvas = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
        output_canvas[binary_mask == 255] = [255, 110, 0]
        
    return output_canvas