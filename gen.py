import numpy as np
import cv2
import os
import random
import json

class SoftFoldGenerator:
    def __init__(self):
        pass

    def _get_curve(self, size, amplitude=5.0):
        t = np.linspace(0, np.random.uniform(1.5, 3.5), size).astype(np.float32)
        phase = np.random.uniform(0, 2*np.pi)
        
        curve = np.sin(t + phase) * amplitude
        curve += np.sin(t * 2.1 + phase) * (amplitude * 0.4)
        
        return curve

    def apply_joined_pages_fold(self, img):
        h, w = img.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        wobble = self._get_curve(h, amplitude=w*0.008).reshape(-1, 1) 
        
        fold_x = (w / 2) + wobble

        line_points = []
        step = 5
        for y in range(0, h, step):
            x = float(fold_x[y, 0])
            line_points.append([x, float(y)])
        fold_lines = [line_points]

        dist_norm = np.abs(grid_x - fold_x) / (w / 2)
        
        amplitude = w * 0.05 
        decay = 10.0
        
        z_map = amplitude * (1 - np.exp(-dist_norm * decay))
        
        y_rel = (grid_y - h/2) / (h/2)
        map_y = grid_y + (z_map * y_rel * 0.35)
        
        direction = np.sign(grid_x - fold_x)
        pinch = np.exp(-dist_norm * 30.0) * 4.0 
        map_x = grid_x - (z_map * direction * 0.12) - (pinch * direction)
        
        left_gradient = 1.0 - (0.05 * np.exp(-dist_norm * 5.0))
        right_gradient = 1.0 - (0.35 * np.exp(-dist_norm * 8.0))
        light_map = np.where(grid_x < fold_x, left_gradient, right_gradient)
        
        light_map = cv2.GaussianBlur(light_map, (5, 5), 0)
        noise = np.random.normal(0, 0.015, (h, w)).astype(np.float32)
        light_map -= (noise * (1.0 - light_map))
        light_map = np.clip(light_map, 0.5, 1.05)
        
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(248,248,248))
        warped_img = warped_img.astype(np.float32) * np.dstack([light_map]*3)
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)
        
        mask_src = np.full((h, w), 255, dtype=np.uint8)
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0) 
        
        return warped_img, warped_mask, fold_lines

    def apply_z_fold(self, img):
        h, w = img.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        base_f1 = w * 0.333
        base_f2 = w * 0.666
        
        wobble1 = self._get_curve(h, amplitude=w*0.007).reshape(-1, 1)
        wobble2 = self._get_curve(h, amplitude=w*0.007).reshape(-1, 1)
        
        f1_map = base_f1 + wobble1
        f2_map = base_f2 + wobble2
        
        line1, line2 = [], []
        step = 5
        for y in range(0, h, step):
            line1.append([float(f1_map[y, 0]), float(y)])
            line2.append([float(f2_map[y, 0]), float(y)])
        fold_lines = [line1, line2]

        d1 = np.abs(grid_x - f1_map)
        d2 = np.abs(grid_x - f2_map)
        
        amp = w * 0.04
        decay_factor = w * 0.1
        
        z1 = amp * (1 - np.exp(-d1 / decay_factor))
        z2 = amp * (1 - np.exp(-d2 / decay_factor))
        
        z_map = np.where(d1 < d2, z1, z2)
        
        y_rel = (grid_y - h/2) / (h/2)
        map_y = grid_y + (z_map * y_rel * 0.3)
        
        nearest_is_1 = d1 < d2
        nearest_fold_map = np.where(nearest_is_1, f1_map, f2_map)
        
        direction = np.sign(grid_x - nearest_fold_map)
        dist_nearest = np.minimum(d1, d2)
        
        pinch = np.exp(-dist_nearest / 30.0) * 4.0
        map_x = grid_x - (z_map * direction * 0.08) - (pinch * direction)

        light_map = np.ones((h, w), dtype=np.float32)
        
        seg1 = 1.0 - (0.05 * np.exp(-(f1_map - grid_x)/10.0))
        shadow_f1 = 1.0 - (0.35 * np.exp(-(grid_x - f1_map)/20.0))
        shadow_f2 = 1.0 - (0.35 * np.exp(-(grid_x - f2_map)/20.0))
        
        light_map = np.select(
            [grid_x < f1_map, (grid_x >= f1_map) & (grid_x < f2_map), grid_x >= f2_map],
            [seg1, shadow_f1, shadow_f2]
        )
        
        light_map = cv2.GaussianBlur(light_map, (5, 5), 0)
        noise = np.random.normal(0, 0.015, (h, w)).astype(np.float32)
        light_map -= (noise * (1.0 - light_map))
        light_map = np.clip(light_map, 0.5, 1.05)

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(248,248,248))
        warped_img = warped_img.astype(np.float32) * np.dstack([light_map]*3)
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)
        
        mask_src = np.full((h, w), 255, dtype=np.uint8)
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return warped_img, warped_mask, fold_lines

    def apply_cross_fold(self, img):
        h, w = img.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        wobble_v = self._get_curve(h, amplitude=w*0.006).reshape(-1, 1)
        cx_map = (w / 2) + wobble_v
        
        wobble_h = self._get_curve(w, amplitude=h*0.006).reshape(1, -1)
        cy_map = (h / 2) + wobble_h

        line_v = []
        step = 5
        for y in range(0, h, step):
            line_v.append([float(cx_map[y, 0]), float(y)])
        
        line_h = []
        for x in range(0, w, step):
            line_h.append([float(x), float(cy_map[0, x])])
            
        fold_lines = [line_v, line_h]
        
        amp_factor = 0.04 
        decay = 10.0
        
        dist_norm_x = np.abs(grid_x - cx_map) / (w / 2)
        z_map_v = (w * amp_factor) * (1 - np.exp(-dist_norm_x * decay))
        
        y_rel = (grid_y - (h/2)) / (h / 2)
        shift_y_from_v = z_map_v * y_rel * 0.35 
        
        direction_x = np.sign(grid_x - cx_map)
        pinch_x = np.exp(-dist_norm_x * 30.0) * 4.0
        shift_x_from_v = -(z_map_v * direction_x * 0.12) - (pinch_x * direction_x)

        dist_norm_y = np.abs(grid_y - cy_map) / (h / 2)
        z_map_h = (h * amp_factor) * (1 - np.exp(-dist_norm_y * decay))
        
        x_rel = (grid_x - (w/2)) / (w / 2)
        shift_x_from_h = z_map_h * x_rel * 0.35
        
        direction_y = np.sign(grid_y - cy_map)
        pinch_y = np.exp(-dist_norm_y * 30.0) * 4.0
        shift_y_from_h = -(z_map_h * direction_y * 0.12) - (pinch_y * direction_y)

        map_x = grid_x + shift_x_from_v + shift_x_from_h
        map_y = grid_y + shift_y_from_v + shift_y_from_h

        left_grad = 1.0 - (0.05 * np.exp(-dist_norm_x * 5.0))
        right_grad = 1.0 - (0.35 * np.exp(-dist_norm_x * 8.0))
        light_v = np.where(grid_x < cx_map, left_grad, right_grad)
        
        top_grad = 1.0 - (0.05 * np.exp(-dist_norm_y * 5.0))
        bot_grad = 1.0 - (0.35 * np.exp(-dist_norm_y * 8.0))
        light_h = np.where(grid_y < cy_map, top_grad, bot_grad)
        
        light_map = light_v * light_h
        
        light_map = cv2.GaussianBlur(light_map, (5, 5), 0)
        noise = np.random.normal(0, 0.015, (h, w)).astype(np.float32)
        light_map -= (noise * (1.0 - light_map))
        light_map = np.clip(light_map, 0.5, 1.05)
        
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(248,248,248))
        
        warped_img = warped_img.astype(np.float32) * np.dstack([light_map]*3)
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)
        
        mask_src = np.full((h, w), 255, dtype=np.uint8)
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return warped_img, warped_mask, fold_lines
    

    def apply_grid_fold(self, img):
        h, w = img.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        base_v1, base_v2, base_v3 = w * 0.25, w * 0.5, w * 0.75
        base_hy = h * 0.5
        
        wb1 = self._get_curve(h, w*0.005).reshape(-1, 1)
        wb2 = self._get_curve(h, w*0.005).reshape(-1, 1)
        wb3 = self._get_curve(h, w*0.005).reshape(-1, 1)
        
        wb_h = self._get_curve(w, h*0.005).reshape(1, -1)
        
        v1 = base_v1 + wb1
        v2 = base_v2 + wb2
        v3 = base_v3 + wb3
        hy = base_hy + wb_h

        l_v1, l_v2, l_v3, l_h = [], [], [], []
        step = 5
        
        for y in range(0, h, step):
            l_v1.append([float(v1[y, 0]), float(y)])
            l_v2.append([float(v2[y, 0]), float(y)])
            l_v3.append([float(v3[y, 0]), float(y)])
            
        for x in range(0, w, step):
            l_h.append([float(x), float(hy[0, x])])
            
        fold_lines = [l_v1, l_v2, l_v3, l_h]
        
        amp_factor = 0.035 
        decay = 10.0
        
        panel_w = w / 4.0
        d1 = np.abs(grid_x - v1) / panel_w
        d2 = np.abs(grid_x - v2) / panel_w
        d3 = np.abs(grid_x - v3) / panel_w
        
        z_v1 = (w * amp_factor) * (1 - np.exp(-d1 * decay))
        z_v2 = (w * amp_factor) * (1 - np.exp(-d2 * decay))
        z_v3 = (w * amp_factor) * (1 - np.exp(-d3 * decay))
        
        z_map_v = np.maximum(np.maximum(z_v1, z_v2), z_v3)
        
        y_rel = (grid_y - (h/2)) / (h / 2)
        shift_y_from_v = z_map_v * y_rel * 0.35 
        
        dist_min_v = np.minimum(np.minimum(d1, d2), d3)
        pinch_v_mag = np.exp(-dist_min_v * 30.0) * 4.0
        
        nearest_v_map = np.select(
            [grid_x < (base_v1+base_v2)/2, grid_x < (base_v2+base_v3)/2, grid_x >= (base_v2+base_v3)/2],
            [v1, v2, v3]
        )
        
        dir_to_nearest_v = np.sign(grid_x - nearest_v_map)
        shift_x_from_v = -(z_map_v * dir_to_nearest_v * 0.1) - (pinch_v_mag * dir_to_nearest_v)

        d_h_norm = np.abs(grid_y - hy) / (h / 2)
        z_map_h = (h * amp_factor) * (1 - np.exp(-d_h_norm * decay))
        
        x_rel = (grid_x - (w/2)) / (w/2)
        shift_x_from_h = z_map_h * x_rel * 0.35
        
        dir_h = np.sign(grid_y - hy)
        pinch_h = np.exp(-d_h_norm * 30.0) * 4.0
        shift_y_from_h = -(z_map_h * dir_h * 0.12) - (pinch_h * dir_h)

        map_x = grid_x + shift_x_from_v + shift_x_from_h
        map_y = grid_y + shift_y_from_v + shift_y_from_h

        def get_fold_light(dist_norm, coord, center_map):
            left = 1.0 - (0.05 * np.exp(-dist_norm * 5.0))
            right = 1.0 - (0.35 * np.exp(-dist_norm * 8.0))
            return np.where(coord < center_map, left, right)

        l1 = get_fold_light(d1, grid_x, v1)
        l2 = get_fold_light(d2, grid_x, v2)
        l3 = get_fold_light(d3, grid_x, v3)
        light_v = l1 * l2 * l3
        
        top_grad = 1.0 - (0.05 * np.exp(-d_h_norm * 5.0))
        bot_grad = 1.0 - (0.35 * np.exp(-d_h_norm * 8.0))
        light_h = np.where(grid_y < hy, top_grad, bot_grad)
        
        light_map = light_v * light_h
        
        light_map = cv2.GaussianBlur(light_map, (5, 5), 0)
        noise = np.random.normal(0, 0.015, (h, w)).astype(np.float32)
        light_map -= (noise * (1.0 - light_map))
        light_map = np.clip(light_map, 0.5, 1.05)
        
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, 
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(248,248,248))
        warped_img = warped_img.astype(np.float32) * np.dstack([light_map]*3)
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)
        
        mask_src = np.full((h, w), 255, dtype=np.uint8)
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return warped_img, warped_mask, fold_lines

    def process_fold(self, img, fold_type):
        if fold_type == 1:
            w_img, w_mask, fold_lines = self.apply_joined_pages_fold(img)
        elif fold_type == 2:
            w_img, w_mask, fold_lines = self.apply_z_fold(img)
        elif fold_type == 3:
            w_img, w_mask, fold_lines = self.apply_cross_fold(img)
        elif fold_type == 4:
            w_img, w_mask, fold_lines = self.apply_grid_fold(img)
        else:
            return img, np.ones(img.shape[:2], dtype=np.float32), []

        h_res, w_res = w_img.shape[:2]
        full_canvas_img = np.zeros((1400, 1400, 3), dtype=np.uint8)
        full_canvas_mask = np.zeros((1400, 1400), dtype=np.uint8)
        
        y_offset = (1400 - h_res) // 2
        x_offset = (1400 - w_res) // 2
        
        full_canvas_img[y_offset:y_offset+h_res, x_offset:x_offset+w_res] = w_img
        full_canvas_mask[y_offset:y_offset+h_res, x_offset:x_offset+w_res] = w_mask
        
        adjusted_folds = []
        for line in fold_lines:
            adj_line = []
            for pt in line:
                adj_line.append([pt[0] + x_offset, pt[1] + y_offset])
            adjusted_folds.append(adj_line)

        return full_canvas_img, full_canvas_mask.astype(np.float32)/255.0, adjusted_folds

    def composite_final(self, doc_img, doc_mask, bg_img):
        if bg_img is None: return doc_img
        h, w = doc_img.shape[:2]
        bg_resized = cv2.resize(bg_img, (w, h))
        mask_soft = cv2.GaussianBlur(doc_mask, (7, 7), 0)
        mask_3ch = np.dstack([mask_soft]*3)
        result = bg_resized.astype(np.float32) * (1.0 - mask_3ch) + doc_img.astype(np.float32) * mask_3ch
        return np.clip(result, 0, 255).astype(np.uint8)

def main():
    input_dir = r"C:\hse\3kursovaya\gen"
    bg_dir = r"C:\hse\3kursovaya\backgrounds"
    out_dir = r"C:\hse\3kursovaya\dataset_ready"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    doc_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]

    bg_img = None
    if bg_files:
        bg_img = cv2.imread(os.path.join(bg_dir, bg_files[0]))

    generator = SoftFoldGenerator()
    types = {1: "1fold", 2: "3fold", 3: "4fold", 4: "8fold"}
    
    crop_offset = 100 

    for f in doc_files:
        source_path = os.path.join(input_dir, f)
        img = cv2.imread(source_path)
        if img is None: continue
        
        abs_scan_path = os.path.abspath(source_path)
        
        img = cv2.resize(img, (600, 850))
        name = os.path.splitext(f)[0]
        
        for t_id, t_name in types.items():
            final, mask, fold_lines = generator.process_fold(img, t_id)
                
            if bg_img is not None:
                bg_large = cv2.resize(bg_img, (1400, 1400))
                final_comp = generator.composite_final(final, mask, bg_large)
            else:
                final_comp = final

            h_c, w_c = final_comp.shape[:2]
            
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            json_vertices = []
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                for point in approx:
                    x, y = point[0]
                    x_final = int(x - crop_offset)
                    y_final = int(y - crop_offset)
                    
                    x_final = max(0, x_final)
                    y_final = max(0, y_final)
                    
                    json_vertices.append([x_final, y_final])

            json_folds = []
            for line in fold_lines:
                processed_line = []
                for pt in line:
                    fx, fy = pt
                    fx_final = fx - crop_offset
                    fy_final = fy - crop_offset
                    
                    processed_line.append([round(fx_final, 1), round(fy_final, 1)])
                json_folds.append(processed_line)

            annotation = {
                "vertices": json_vertices,
                "folds": json_folds,
                "reference": name,
                "scan_image_path": abs_scan_path,
                "folding": t_name,
            }

            final_crop = final_comp[crop_offset:h_c-crop_offset, crop_offset:w_c-crop_offset]
            
            output_base_name = f"{name}_{t_name}"
            output_img_path = os.path.join(out_dir, f"{output_base_name}.jpg")
            output_json_path = os.path.join(out_dir, f"{output_base_name}.json")
            
            cv2.imwrite(output_img_path, final_crop)
            
            with open(output_json_path, 'w', encoding='utf-8') as jf:
                json.dump(annotation, jf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()