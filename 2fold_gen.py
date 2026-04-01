import numpy as np
import cv2
import os
import random
import json

class NaturalFoldGenerator:
    def __init__(self):
        self.target_w = 3024
        self.target_h = 4032
        self.margin = 200

    def _get_curve(self, size, amplitude=1.0):
        t = np.linspace(0, np.random.uniform(1.0, 2.0), size).astype(np.float32)
        phase = np.random.uniform(0, 2*np.pi)
        curve = np.sin(t + phase) * amplitude
        return curve

    def apply_2fold(self, img, mode="up"):
        h, w = img.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        offset_center = random.uniform(-h * 0.1, h * 0.1)
        slant_factor = random.uniform(-h * 0.05, h * 0.05)
        x_norm = (grid_x - w/2) / w
        base_fold_y = (h / 2) + offset_center + (slant_factor * x_norm)
        wobble = self._get_curve(w, amplitude=h*0.001).reshape(1, -1) 
        fold_y = base_fold_y + wobble

        line_points = []
        step = 10 
        for x in range(0, w, step):
            y_val = float(fold_y[h//2, x]) 
            line_points.append([float(x), y_val])
        fold_lines = [line_points]

        dist_from_fold = grid_y - fold_y
        dist_norm = np.abs(dist_from_fold) / (h / 2)
        is_down = (mode == "down")
        x_centered = (grid_x - w/2) / (w/2)

        if is_down:
            narrowing_power = random.uniform(0.08, 0.12) 
            influence_width = 4.0 
            
            shrink = 1.0 - (narrowing_power * np.exp(-dist_norm * influence_width))
            
            map_x = w/2 + (grid_x - w/2) / shrink
            
            map_y = grid_y + (dist_from_fold * 0.15 * np.exp(-dist_norm * 2.5))
            
        else:
            height_profile = 1.0 + self._get_curve(w, amplitude=0.4).reshape(1, -1)
            
            base_amplitude_z = h * random.uniform(0.03, 0.05) 
            
            amplitude_z_map = base_amplitude_z * height_profile 
            
            decay = 12.0
            z_map = amplitude_z_map * (1 - np.exp(-dist_norm * decay))
            
            direction = np.sign(dist_from_fold)
            
            map_x = grid_x + (z_map * x_centered * 0.3)
            
            pinch = np.exp(-dist_norm * 100.0) * 8.0 
            map_y = grid_y - (z_map * direction * 0.1) - (pinch * direction)

        shadow_depth_top = random.uniform(0.02, 0.08)
        shadow_depth_bottom = random.uniform(0.1, 0.4)
        brightness_base = random.uniform(0.98, 1.03)
        min_light_val = random.uniform(0.55, 0.8)
        shadow_width_top = random.uniform(3.0, 8.0)
        shadow_width_bottom = random.uniform(5.0, 12.0)
        
        top_gradient = brightness_base - (shadow_depth_top * np.exp(-dist_norm * shadow_width_top))
        bottom_gradient = brightness_base - (shadow_depth_bottom * np.exp(-dist_norm * shadow_width_bottom))
        
        if is_down:
            light_map = np.where(dist_from_fold < 0, bottom_gradient, top_gradient)
        else:
            light_map = np.where(dist_from_fold < 0, top_gradient, bottom_gradient)
        
        light_map = cv2.GaussianBlur(light_map, (15, 15), 0)
        light_map = np.clip(light_map, min_light_val, 1.05)
        
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(248,248,248))
        warped_img = warped_img.astype(np.float32) * np.dstack([light_map]*3)
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)
        
        mask_src = np.full((h, w), 255, dtype=np.uint8)
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return warped_img, warped_mask, fold_lines

    def process(self, img):
        mode = random.choice(["up", "down"])
        w_img, w_mask, fold_lines = self.apply_2fold(img, mode=mode)
        h_res, w_res = w_img.shape[:2]
        canvas_h, canvas_w = self.target_h + 2*self.margin, self.target_w + 2*self.margin
        
        full_canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        full_canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        
        y_offset = (canvas_h - h_res) // 2
        x_offset = (canvas_w - w_res) // 2
        
        full_canvas_img[y_offset:y_offset+h_res, x_offset:x_offset+w_res] = w_img
        full_canvas_mask[y_offset:y_offset+h_res, x_offset:x_offset+w_res] = w_mask
        
        adjusted_folds = []
        for line in fold_lines:
            adj_line = [[pt[0] + x_offset, pt[1] + y_offset] for pt in line]
            adjusted_folds.append(adj_line)

        return full_canvas_img, full_canvas_mask.astype(np.float32)/255.0, adjusted_folds

    def composite_final(self, doc_img, doc_mask, bg_img):
        h, w = doc_img.shape[:2]
        bg_resized = cv2.resize(bg_img, (w, h))
        mask_soft = cv2.GaussianBlur(doc_mask, (15, 15), 0)
        mask_3ch = np.dstack([mask_soft]*3)
        result = bg_resized.astype(np.float32) * (1.0 - mask_3ch) + doc_img.astype(np.float32) * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)

def main():
    input_dir = r"C:\hse\3kursovaya\gen"
    bg_dir = r"C:\hse\3kursovaya\backgrounds"
    out_dir = r"C:\hse\3kursovaya\dataset_ready_2fold"
    
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    doc_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]
    generator = NaturalFoldGenerator()

    for f in doc_files:
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None: continue
        img = cv2.resize(img, (2200, 3100))
        name = os.path.splitext(f)[0]
        
        final_canvas, mask, fold_lines = generator.process(img)
        if bg_files:
            bg_img = cv2.imread(os.path.join(bg_dir, random.choice(bg_files)))
            final_comp = generator.composite_final(final_canvas, mask, bg_img)
        else:
            final_comp = final_canvas

        OFFSET = generator.margin
        final_crop = final_comp[OFFSET:OFFSET+4032, OFFSET:OFFSET+3024]
        mask_crop = mask[OFFSET:OFFSET+4032, OFFSET:OFFSET+3024]

        mask_uint8 = (mask_crop * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        json_vertices = []
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
            for point in approx: json_vertices.append([int(point[0][0]), int(point[0][1])])

        json_folds = []
        for line in fold_lines:
            json_folds.append([[round(pt[0]-OFFSET, 1), round(pt[1]-OFFSET, 1)] for pt in line])

        annotation = {
            "vertices": json_vertices,
            "folds": json_folds,
            "folding": "2fold",
            "resolution": [3024, 4032]
        }

        cv2.imwrite(os.path.join(out_dir, f"{name}_2fold.jpg"), final_crop)
        with open(os.path.join(out_dir, f"{name}_2fold.json"), 'w', encoding='utf-8') as jf:
            json.dump(annotation, jf, indent=2)

if __name__ == "__main__":
    main()