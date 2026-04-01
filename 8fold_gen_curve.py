import numpy as np
import cv2
import os
import random
import json

class EightFoldGenerator:
    def __init__(self):
        self.target_w = 3024
        self.target_h = 4032
        self.margin = 450 

    def _get_curve(self, size, amplitude=1.0):
        t = np.linspace(0, np.random.uniform(1.0, 2.0), size).astype(np.float32)
        phase = np.random.uniform(0, 2*np.pi)
        curve = np.sin(t + phase) * amplitude
        return curve

    def _generate_lighting(self, dist_norm, dist_from_fold, is_down, strength=1.0):
        sh_t = random.uniform(0.02, 0.05) * strength
        sh_b = random.uniform(0.15, 0.35) * strength
        brightness = random.uniform(0.98, 1.02)
        grad_t = brightness - (sh_t * np.exp(-dist_norm * 6.0))
        grad_b = brightness - (sh_b * np.exp(-dist_norm * 10.0))
        if is_down:
            return np.where(dist_from_fold < 0, grad_b, grad_t)
        return np.where(dist_from_fold < 0, grad_t, grad_b)

    def apply_8fold(self, full_canvas, sheet_rect):
        h_canv, w_canv = full_canvas.shape[:2]
        grid_y, grid_x = np.mgrid[0:h_canv, 0:w_canv].astype(np.float32)
        sx, sy, sw, sh = sheet_rect
        cx, cy = int(sx + sw // 2), int(sy + sh // 2)

        total_dx = np.zeros_like(grid_x)
        total_dy = np.zeros_like(grid_y)
        combined_light = np.ones((h_canv, w_canv), dtype=np.float32)

        # 1. ГЕОМЕТРИЯ ЛИНИЙ (Смещение и Наклон)
        x_norm = (grid_x[0, :] - cx) / sw
        y_norm = (grid_y[:, 0] - cy) / sh

        # L1: Главная горизонталь
        h1_off = random.uniform(-sh*0.05, sh*0.05)
        h1_slant = random.uniform(-sh*0.02, sh*0.02)
        main_h_y = (cy + h1_off + (h1_slant * x_norm) + self._get_curve(w_canv, 0.001*sh)).reshape(1, -1)
        
        # L2: Вертикальный разлом
        v_off = random.uniform(-sw*0.04, sw*0.04)
        v_slant = random.uniform(-sw*0.02, sw*0.02)
        v_line_x = (cx + v_off + (v_slant * y_norm) + self._get_curve(h_canv, 0.001*sw)).reshape(-1, 1)

        # L3: Вспомогательные горизонтали (Top и Bottom)
        h3t_off = random.uniform(-sh*0.03, sh*0.03)
        h3t_slant = random.uniform(-sh*0.015, sh*0.015)
        h_sub_top_y = (sy + sh/4 + h3t_off + (h3t_slant * x_norm) + self._get_curve(w_canv, 0.001*sh)).reshape(1, -1)

        h3b_off = random.uniform(-sh*0.03, sh*0.03)
        h3b_slant = random.uniform(-sh*0.015, sh*0.015)
        h_sub_bot_y = (sy + 3*sh/4 + h3b_off + (h3b_slant * x_norm) + self._get_curve(w_canv, 0.001*sh)).reshape(1, -1)

        # 2. ОСНОВНАЯ ГОРИЗОНТАЛЬ (L1)
        h1_mode = random.choice(["up", "down"])
        is_h1_down = (h1_mode == "down")
        dist_h1 = grid_y - main_h_y
        dist_h1_norm = np.clip(np.abs(dist_h1) / (sh / 2), 0, 1.2)
        h1_inf = np.exp(-dist_h1_norm * 4.0)
        
        if is_h1_down:
            total_dx += (grid_x - cx) * (0.1 * h1_inf)
            total_dy += dist_h1 * (0.15 * h1_inf)
        else:
            z_m = (sh * 0.04) * (1 - np.exp(-dist_h1_norm * 10.0))
            total_dx += z_m * ((grid_x - cx) / (sw/2)) * 0.3 * h1_inf
            total_dy -= z_m * np.sign(dist_h1) * 0.1 * h1_inf
        combined_light *= self._generate_lighting(dist_h1_norm, dist_h1, is_h1_down)

        # 3. ИНВЕРСИЯ КВАДРАНТОВ (L2 и L3)
        v_top_mode = "up" if random.random() > 0.5 else "down"
        v_bot_mode = "down" if v_top_mode == "up" else "up"
        
        l3_tl_mode = "up" if random.random() > 0.5 else "down"
        l3_tr_mode = "down" if l3_tl_mode == "up" else "up"
        l3_bl_mode = "down" if l3_tl_mode == "up" else "up"
        l3_br_mode = "up" if l3_tl_mode == "up" else "down"

        mask_top = grid_y < main_h_y
        mask_bot = ~mask_top
        mask_left = grid_x < v_line_x
        mask_right = ~mask_left

        quads = [
            (mask_top & mask_left, v_top_mode, l3_tl_mode, h_sub_top_y),
            (mask_top & mask_right, v_top_mode, l3_tr_mode, h_sub_top_y),
            (mask_bot & mask_left, v_bot_mode, l3_bl_mode, h_sub_bot_y),
            (mask_bot & mask_right, v_bot_mode, l3_br_mode, h_sub_bot_y)
        ]

        for q_mask, v_m, h3_m, h3_y in quads:
            # Вертикаль (L2)
            is_v_down = (v_m == "down")
            dist_v = grid_x - v_line_x
            dist_v_norm = np.clip(np.abs(dist_v) / (sw / 4), 0, 1.2)
            long_inf = np.clip(np.abs(grid_y - main_h_y) / (sh/2), 0, 1.0)
            v_inf = np.exp(-dist_v_norm * 5.0) * long_inf
            
            pwr_v = 0.07
            if is_v_down:
                total_dx = np.where(q_mask, total_dx + dist_v * pwr_v * v_inf, total_dx)
                total_dy = np.where(q_mask, total_dy + (grid_y - main_h_y) * 0.08 * v_inf, total_dy)
            else:
                p_f = 1.0 / (1.0 + pwr_v * v_inf)
                total_dx = np.where(q_mask, total_dx + (grid_x - cx) * (p_f - 1.0), total_dx)
                total_dy = np.where(q_mask, total_dy + (grid_y - main_h_y) * (p_f - 1.0) * 0.5, total_dy)
            combined_light = np.where(q_mask, combined_light * self._generate_lighting(dist_v_norm, dist_v, is_v_down), combined_light)

            # Горизонталь 3 степени (L3)
            is_h3_down = (h3_m == "down")
            dist_h3 = grid_y - h3_y
            dist_h3_norm = np.clip(np.abs(dist_h3) / (sh / 8), 0, 1.2)
            x_inf = np.clip(np.abs(grid_x - v_line_x) / (sw/4), 0, 1.0)
            h3_inf = np.exp(-dist_h3_norm * 6.5) * x_inf * long_inf
            
            pwr_h3 = 0.05
            if is_h3_down:
                total_dx = np.where(q_mask, total_dx + (grid_x - cx) * pwr_h3 * h3_inf, total_dx)
                total_dy = np.where(q_mask, total_dy + dist_h3 * pwr_h3 * h3_inf, total_dy)
            else:
                p_f = 1.0 / (1.0 + pwr_h3 * h3_inf)
                total_dx = np.where(q_mask, total_dx + (grid_x - cx) * (p_f - 1.0), total_dx)
                total_dy = np.where(q_mask, total_dy + (grid_y - main_h_y) * (p_f - 1.0) * 0.4, total_dy)
            combined_light = np.where(q_mask, combined_light * self._generate_lighting(dist_h3_norm, dist_h3, is_h3_down), combined_light)

        # РЕНДЕРИНГ
        map_x, map_y = grid_x + total_dx, grid_y + total_dy
        combined_light = cv2.GaussianBlur(combined_light, (15, 15), 0).clip(0.5, 1.05)
        warped_img = cv2.remap(full_canvas, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderValue=(0,0,0))
        warped_img = (warped_img.astype(np.float32) * np.dstack([combined_light]*3)).clip(0, 255).astype(np.uint8)
        mask_src = np.zeros((h_canv, w_canv), dtype=np.uint8); mask_src[sy:sy+sh, sx:sx+sw] = 255
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0)

        # JSON Линии (Стык в точках пересечения)
        fold_lines = []
        # L1
        fold_lines.append([[float(x), float(main_h_y[0, int(x)])] for x in range(int(sx), int(sx+sw), 40)])
        # L2 (Разделена пересечением с L1)
        v_idx = int(np.clip(cx + v_off, 0, w_canv-1))
        inter_y = main_h_y[0, v_idx]
        fold_lines.append([[float(v_line_x[int(y), 0]), float(y)] for y in range(int(sy), int(inter_y), 40)])
        fold_lines.append([[float(v_line_x[int(y), 0]), float(y)] for range_y in [range(int(inter_y), int(sy+sh), 40)] for y in range_y])
        # L3 (Разделены пересечением с L2)
        h_idx = int(np.clip(cy + h1_off, 0, h_canv-1))
        for h_y in [h_sub_top_y, h_sub_bot_y]:
            inter_x = v_line_x[int(h_y[0, v_idx] if v_idx < h_canv else 0), 0]
            fold_lines.append([[float(x), float(h_y[0, int(x)])] for x in range(int(sx), int(inter_x), 40)])
            fold_lines.append([[float(x), float(h_y[0, int(x)])] for x in range(int(inter_x), int(sx+sw), 40)])

        return warped_img, warped_mask, fold_lines

    def process(self, img):
        img_h, img_w = img.shape[:2]
        canvas_h, canvas_w = self.target_h + 2*self.margin, self.target_w + 2*self.margin
        full_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        y_off, x_off = (canvas_h - img_h)//2, (canvas_w - img_w)//2
        full_canvas[y_off:y_off+img_h, x_off:x_off+img_w] = img
        warped_img, warped_mask, fold_lines = self.apply_8fold(full_canvas, (x_off, y_off, img_w, img_h))
        return warped_img, warped_mask.astype(np.float32)/255.0, fold_lines

    def composite_final(self, doc_img, doc_mask, bg_img):
        h, w = doc_img.shape[:2]
        bg = cv2.resize(bg_img, (w, h))
        mask_soft = cv2.GaussianBlur(doc_mask, (11, 11), 0)
        mask_3ch = np.dstack([mask_soft]*3)
        res = bg.astype(np.float32) * (1.0 - mask_3ch) + doc_img.astype(np.float32) * mask_3ch
        return res.clip(0, 255).astype(np.uint8)

def main():
    input_dir = r"C:\hse\3kursovaya\gen"; bg_dir = r"C:\hse\3kursovaya\backgrounds"; out_dir = r"C:\hse\3kursovaya\dataset_ready_8fold"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    doc_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]
    generator = EightFoldGenerator()
    for f_name in doc_files:
        img = cv2.imread(os.path.join(input_dir, f_name)); 
        if img is None: continue
        img = cv2.resize(img, (2200, 3100))
        canvas, mask, fold_lines = generator.process(img)
        if bg_files:
            bg = cv2.imread(os.path.join(bg_dir, random.choice(bg_files)))
            final_img = generator.composite_final(canvas, mask, bg)
        else: final_img = canvas
        off = generator.margin; final_crop = final_img[off:off+4032, off:off+3024]
        mask_crop = (mask[off:off+4032, off:off+3024] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vertices = []
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
            vertices = [[int(p[0][0]), int(p[0][1])] for p in approx]
        json_folds = [[[round(pt[0]-off, 1), round(pt[1]-off, 1)] for pt in line] for line in fold_lines]
        name = os.path.splitext(f_name)[0]
        anno = {"vertices": vertices, "folds": json_folds, "folding": "8fold", "resolution": [3024, 4032]}
        cv2.imwrite(os.path.join(out_dir, f"{name}_8fold.jpg"), final_crop)
        with open(os.path.join(out_dir, f"{name}_8fold.json"), 'w', encoding='utf-8') as jf:
            json.dump(anno, jf, indent=2)
        print(f"Generated Realistic 8fold: {name}")

if __name__ == "__main__":
    main()