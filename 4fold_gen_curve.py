import numpy as np
import cv2
import os
import random
import json

class FourFoldGenerator:
    def __init__(self):
        self.target_w = 3024
        self.target_h = 4032
        self.margin = 400 

    def _get_curve(self, size, amplitude=1.0):
        t = np.linspace(0, np.random.uniform(1.0, 2.0), size).astype(np.float32)
        phase = np.random.uniform(0, 2*np.pi)
        curve = np.sin(t + phase) * amplitude
        return curve

    def _generate_lighting(self, dist_norm, dist_from_fold, is_down, strength=1.0, 
                          sh_t_val=0.04, sh_b_val=0.25, decay_t=6.0, decay_b=10.0):
        brightness = random.uniform(0.97, 1.03)
        grad_t = brightness - (sh_t_val * np.exp(-dist_norm * decay_t))
        grad_b = brightness - (sh_b_val * np.exp(-dist_norm * decay_b))
        if is_down:
            return np.where(dist_from_fold < 0, grad_b, grad_t)
        return np.where(dist_from_fold < 0, grad_t, grad_b)

    def apply_4fold(self, full_canvas, sheet_rect):
        h_canv, w_canv = full_canvas.shape[:2]
        grid_y, grid_x = np.mgrid[0:h_canv, 0:w_canv].astype(np.float32)
        
        sx, sy, sw, sh = sheet_rect
        cx, cy = int(sx + sw // 2), int(sy + sh // 2)

        # 1. ГЕНЕРАЦИЯ ГЕОМЕТРИИ ЛИНИЙ (Смещение и Наклон)
        # Горизонтальная магистраль
        off_h = random.uniform(-sh * 0.04, sh * 0.04) # Смещение вверх/вниз
        slant_h = random.uniform(-sh * 0.02, sh * 0.02) # Наклон (перепад высот)
        x_norm = (grid_x[0, :] - cx) / sw
        main_y_arr = (cy + off_h + (slant_h * x_norm) + self._get_curve(w_canv, 0.001*sh)).reshape(1, -1)
        
        # Вертикальный разлом
        off_v = random.uniform(-sw * 0.04, sw * 0.04) # Смещение влево/вправо
        slant_v = random.uniform(-sw * 0.02, sw * 0.02) # Наклон
        y_norm = (grid_y[:, 0] - cy) / sh
        branch_x_arr = (cx + off_v + (slant_v * y_norm) + self._get_curve(h_canv, 0.001*sw)).reshape(-1, 1)

        total_dx = np.zeros_like(grid_x)
        total_dy = np.zeros_like(grid_y)
        combined_light = np.ones((h_canv, w_canv), dtype=np.float32)

        # ПАРАМЕТРЫ СВЕТА
        sh_t_rand = random.uniform(0.015, 0.05)
        sh_b_rand = random.uniform(0.15, 0.4)
        blur_k = random.choice([13, 15, 17])

        # ВЫБОР РЕЖИМА ГЕНЕРАЦИИ
        rand_val = random.random()
        if rand_val < 0.20: mode_type = "HALF_FLAT"
        elif rand_val < 0.60: mode_type = "OFFSET_SOFT"
        else: mode_type = "STRICT_CENTER"

        # 2. ЦЕНТРАЛЬНЫЙ СГИБ
        dist_h = grid_y - main_y_arr
        dist_h_norm = np.clip(np.abs(dist_h) / (sh / 2), 0, 1.5)
        
        if mode_type == "HALF_FLAT":
            flat_side = random.choice(["top", "bottom"])
            raised_mask = (dist_h > 0) if flat_side == "top" else (dist_h < 0)
            expand_pwr = random.uniform(0.08, 0.14) 
            perspective_factor = 1.0 / (1.0 + expand_pwr * dist_h_norm)
            total_dx = np.where(raised_mask, (grid_x - cx) * (perspective_factor - 1.0), total_dx)
            total_dy = np.where(raised_mask, (grid_y - cy) * (perspective_factor - 1.0) * 0.6, total_dy)
            combined_light *= self._generate_lighting(dist_h_norm, dist_h, is_down=True, strength=1.3)
        else:
            h_mode = random.choice(["up", "down"])
            is_main_down = (h_mode == "down")
            h_influence = np.exp(-dist_h_norm * random.uniform(2.8, 4.2))
            if is_main_down:
                total_dx += (grid_x - cx) * (random.uniform(0.08, 0.12) * h_influence)
                total_dy += dist_h * (random.uniform(0.12, 0.18) * h_influence)
            else:
                amp_z = sh * random.uniform(0.04, 0.055)
                z_m = amp_z * (1 - np.exp(-dist_h_norm * 10.0))
                total_dx += z_m * ((grid_x - cx) / (sw/2)) * 0.35 * (0.4 + 0.6 * h_influence)
                total_dy -= z_m * np.sign(dist_h) * 0.1 * h_influence
            combined_light *= self._generate_lighting(dist_h_norm, dist_h, is_main_down, sh_t_val=sh_t_rand, sh_b_val=sh_b_rand)

        # 3. ВЕРТИКАЛЬНЫЕ КРЫЛЬЯ (используем фактическую линию branch_x_arr для масок)
        v_top_mode = random.choice(["up", "down"])
        v_bottom_mode = "down" if v_top_mode == "up" else "up"
        dist_v = grid_x - branch_x_arr
        dist_v_norm = np.clip(np.abs(dist_v) / (sw / 2), 0, 1.5)

        # Настройки интенсивности
        v_power = random.uniform(0.08, 0.14) if mode_type == "STRICT_CENTER" else random.uniform(0.05, 0.10)
        v_decay = np.exp(-dist_v_norm * (4.0 if mode_type == "STRICT_CENTER" else 5.0)) 

        # Используем main_y_arr для разделения на Верх и Низ
        mask_top = grid_y < main_y_arr
        mask_bot = ~mask_top

        for i, (q_mask, v_mode) in enumerate(zip([mask_top, mask_bot], [v_top_mode, v_bottom_mode])):
            is_side_down = (v_mode == "down")
            current_side = "top" if i == 0 else "bottom"
            
            # Влияние плавно растет от линии горизонтального сгиба
            y_dist_from_h = np.abs(grid_y - main_y_arr) / (sh/2)
            if mode_type == "STRICT_CENTER":
                long_inf = np.clip(0.4 + 0.6 * y_dist_from_h, 0.4, 1.0)
            else:
                long_inf = np.clip(y_dist_from_h, 0, 1.0)
            
            side_mult = 0.2 if (mode_type == "HALF_FLAT" and current_side == flat_side) else 1.0
            local_inf = v_decay * long_inf * side_mult

            if is_side_down:
                total_dx = np.where(q_mask, total_dx + dist_v * v_power * local_inf, total_dx)
                total_dy = np.where(q_mask, total_dy + (grid_y - cy) * (v_power*0.7) * local_inf, total_dy)
            else:
                expand_v = (v_power * 1.1) * local_inf
                total_dx = np.where(q_mask, total_dx - dist_v * expand_v, total_dx)
                total_dy = np.where(q_mask, total_dy - (grid_y - cy) * (expand_v * 0.8), total_dy)

            l_strength = 1.4 if mode_type == "STRICT_CENTER" else 1.1
            side_light = self._generate_lighting(dist_v_norm, dist_v, is_side_down, strength=l_strength,
                                                 sh_t_val=sh_t_rand*0.7, sh_b_val=sh_b_rand*0.7)
            fade_light = 1.0 + (side_light - 1.0) * (long_inf * side_mult)
            combined_light = np.where(q_mask, combined_light * fade_light, combined_light)

        # 4. РЕНДЕРИНГ
        map_x, map_y = grid_x + total_dx, grid_y + total_dy
        combined_light = cv2.GaussianBlur(combined_light, (blur_k, blur_k), 0).clip(0.5, 1.05)
        warped_img = cv2.remap(full_canvas, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderValue=(0,0,0))
        warped_img = (warped_img.astype(np.float32) * np.dstack([combined_light]*3)).clip(0, 255).astype(np.uint8)
        mask_src = np.zeros((h_canv, w_canv), dtype=np.uint8); mask_src[sy:sy+sh, sx:sx+sw] = 255
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0)

        # JSON Линии
        fold_lines = []
        fold_lines.append([[float(x), float(main_y_arr[0, int(x)])] for x in range(int(sx), int(sx+sw), 40)])
        # Точка пересечения для разделения вертикали
        # Находим примерную координату Y магистрали в месте прохождения вертикального разлома
        # Это нужно, чтобы линии в JSON тоже были под углом и стыковались
        v_idx = int(cx + off_v)
        v_idx = np.clip(v_idx, 0, w_canv-1)
        intersection_y = main_y_arr[0, v_idx]
        
        fold_lines.append([[float(branch_x_arr[int(y), 0]), float(y)] for y in range(int(sy), int(intersection_y), 40)])
        fold_lines.append([[float(branch_x_arr[int(y), 0]), float(y)] for y in range(int(intersection_y), int(sy+sh), 40)])
        
        return warped_img, warped_mask, fold_lines

    def process(self, img):
        img_h, img_w = img.shape[:2]
        canvas_h, canvas_w = self.target_h + 2*self.margin, self.target_w + 2*self.margin
        full_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        y_off, x_off = (canvas_h - img_h)//2, (canvas_w - img_w)//2
        full_canvas[y_off:y_off+img_h, x_off:x_off+img_w] = img
        warped_img, warped_mask, fold_lines = self.apply_4fold(full_canvas, (x_off, y_off, img_w, img_h))
        return warped_img, warped_mask.astype(np.float32)/255.0, fold_lines

    def composite_final(self, doc_img, doc_mask, bg_img):
        h, w = doc_img.shape[:2]
        bg = cv2.resize(bg_img, (w, h))
        mask_soft = cv2.GaussianBlur(doc_mask, (11, 11), 0)
        mask_3ch = np.dstack([mask_soft]*3)
        res = bg.astype(np.float32) * (1.0 - mask_3ch) + doc_img.astype(np.float32) * mask_3ch
        return res.clip(0, 255).astype(np.uint8)

def main():
    input_dir = r"C:\hse\3kursovaya\gen"; bg_dir = r"C:\hse\3kursovaya\backgrounds"; out_dir = r"C:\hse\3kursovaya\dataset_ready_4fold"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    doc_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]
    generator = FourFoldGenerator()
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
        anno = {"vertices": vertices, "folds": json_folds, "folding": "4fold", "resolution": [3024, 4032]}
        cv2.imwrite(os.path.join(out_dir, f"{name}_4fold.jpg"), final_crop)
        with open(os.path.join(out_dir, f"{name}_4fold.json"), 'w', encoding='utf-8') as jf:
            json.dump(anno, jf, indent=2)

if __name__ == "__main__":
    main()