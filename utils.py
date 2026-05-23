import cv2

def fit_image_to_viewport(img, max_h=360):
    raw_h, raw_w = img.shape[:2]
    if raw_h <= max_h:
        return img, 1.0
    scale = max_h / raw_h
    new_w = int(raw_w * scale)
    scaled_img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
    return scaled_img, scale