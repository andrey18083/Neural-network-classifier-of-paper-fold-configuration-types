import numpy as np
import cv2
import os
import random
import json

class TriFoldGenerator:
    def __init__(self):
        self.target_w = 3024
        self.target_h = 4032
        self.margin = 350 

    def _get_curve(self, size, amplitude=1.0):
        t = np.linspace(0, np.random.uniform(1.0, 2.0), size).astype(np.float32)
        phase = np.random.uniform(0, 2*np.pi)
        curve = np.sin(t + phase) * amplitude
        return curve

    def _generate_lighting(self, dist_norm, dist_from_fold, is_down, strength=1.0):
        sh_t = random.uniform(0.02, 0.05) * strength
        sh_b = random.uniform(0.15, 0.4) * strength
        brightness = random.uniform(0.98, 1.02)
        grad_t = brightness - (sh_t * np.exp(-dist_norm * random.uniform(3, 7)))
        grad_b = brightness - (sh_b * np.exp(-dist_norm * random.uniform(5, 12)))
        if is_down:
            return np.where(dist_from_fold < 0, grad_b, grad_t)
        return np.where(dist_from_fold < 0, grad_t, grad_b)

    def apply_folds(self, full_canvas, sheet_rect):
        canvas_h, canvas_w = full_canvas.shape[:2]
        grid_y, grid_x = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
        
        sx, sy, sw, sh = sheet_rect
        cx, cy = sx + sw/2, sy + sh/2 
        
        # 1. ГЕНЕРАЦИЯ ГЕОМЕТРИИ ЛИНИЙ (Смещение и Наклон)
        x_norm = (grid_x[0, :] - cx) / sw
        
        # Линия 1 (~1/3 высоты)
        f1_off = random.uniform(-sh*0.05, sh*0.05)
        f1_slant = random.uniform(-sh*0.02, sh*0.02)
        f1_y = (sy + sh/3 + f1_off + (f1_slant * x_norm) + self._get_curve(canvas_w, 0.001*sh)).reshape(1, -1)
        
        # Линия 2 (~2/3 высоты)
        f2_off = random.uniform(-sh*0.05, sh*0.05)
        f2_slant = random.uniform(-sh*0.02, sh*0.02)
        f2_y = (sy + 2*sh/3 + f2_off + (f2_slant * x_norm) + self._get_curve(canvas_w, 0.001*sh)).reshape(1, -1)

        map_x = grid_x.copy()
        map_y = grid_y.copy()
        combined_light = np.ones((canvas_h, canvas_w), dtype=np.float32)
        
        # Шанс 15% на режим "Лоток"
        if random.random() < 0.15:
            expand_w = random.uniform(0.08, 0.14) 
            expand_h = random.uniform(0.03, 0.06) 
            
            for i, f_y in enumerate([f1_y, f2_y]):
                dist_f = grid_y - f_y
                dist_n = np.clip(np.abs(dist_f) / (sh / 3), 0, 1.2)
                side_mask = (dist_f < 0) if i == 0 else (dist_f > 0)
                
                # Коэффициент расширения к камере
                p_factor = 1.0 / (1.0 + expand_w * dist_n)
                
                map_x = np.where(side_mask, cx + (grid_x - cx) * p_factor, map_x)
                
                # Удлинение по Y
                p_factor_y = 1.0 / (1.0 + expand_h * dist_n)
                map_y = np.where(side_mask, f_y + (grid_y - f_y) * p_factor_y, map_y)
                
                combined_light *= self._generate_lighting(dist_n, dist_f, is_down=False, strength=1.4)
        else:
            # СТАНДАРТНЫЙ C/Z-fold (85%)
            total_dx = np.zeros_like(grid_x)
            total_dy = np.zeros_like(grid_y)
            for f_y in [f1_y, f2_y]:
                mode = random.choice(["up", "down"])
                dist_f = grid_y - f_y
                dist_n = np.abs(dist_f) / (sh / 3)
                is_d = (mode == "down")
                if is_d:
                    inf = np.exp(-dist_n * 4.0)
                    total_dx += (grid_x - cx) * (random.uniform(0.07, 0.1) * inf)
                    total_dy += dist_f * 0.1 * np.exp(-dist_n * 2.5)
                else:
                    amp_z = (sh * random.uniform(0.02, 0.035)) * (1.0 + self._get_curve(canvas_w, 0.4).reshape(1, -1))
                    z_m = amp_z * (1 - np.exp(-dist_n * 12.0))
                    total_dx += (z_m * ((grid_x - cx) / (sw/2)) * 0.3)
                    total_dy -= (z_m * np.sign(dist_f) * 0.1) + (np.exp(-dist_n * 80.0) * 5.0 * np.sign(dist_f))
                combined_light *= self._generate_lighting(dist_n, dist_f, is_d)
            map_x += total_dx
            map_y += total_dy

        combined_light = cv2.GaussianBlur(combined_light, (15, 15), 0).clip(0.5, 1.05)
        warped_img = cv2.remap(full_canvas, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderValue=(0,0,0))
        warped_img = (warped_img.astype(np.float32) * np.dstack([combined_light]*3)).clip(0, 255).astype(np.uint8)
        
        mask_src = np.zeros((canvas_h, canvas_w), dtype=np.uint8); mask_src[sy:sy+sh, sx:sx+sw] = 255
        warped_mask = cv2.remap(mask_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0)
        
        json_folds = []
        for f_y in [f1_y, f2_y]:
            json_folds.append([[float(x), float(f_y[0, int(x)])] for x in range(0, canvas_w, 30)])

        return warped_img, warped_mask, json_folds

    def process(self, img):
        img_h, img_w = img.shape[:2]
        canvas_h, canvas_w = self.target_h + 2*self.margin, self.target_w + 2*self.margin
        full_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        y_off, x_off = (canvas_h - img_h)//2, (canvas_w - img_w)//2
        full_canvas[y_off:y_off+img_h, x_off:x_off+img_w] = img
        warped_img, warped_mask, fold_lines = self.apply_folds(full_canvas, (x_off, y_off, img_w, img_h))
        return warped_img, warped_mask.astype(np.float32)/255.0, fold_lines

    def composite_final(self, doc_img, doc_mask, bg_img):
        h, w = doc_img.shape[:2]
        bg = cv2.resize(bg_img, (w, h))
        mask_soft = cv2.GaussianBlur(doc_mask, (9, 9), 0)
        mask_3ch = np.dstack([mask_soft]*3)
        res = bg.astype(np.float32) * (1.0 - mask_3ch) + doc_img.astype(np.float32) * mask_3ch
        return res.clip(0, 255).astype(np.uint8)

def main():
    input_dir = r"C:\hse\3kursovaya\gen"; bg_dir = r"C:\hse\3kursovaya\backgrounds"; out_dir = r"C:\hse\3kursovaya\dataset_ready_3fold"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    doc_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]
    generator = TriFoldGenerator()
    for f_name in doc_files:
        img = cv2.imread(os.path.join(input_dir, f_name))
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
            approx = cv2.approxPolyDP(cnt, 0.004 * cv2.arcLength(cnt, True), True)
            vertices = [[int(p[0][0]), int(p[0][1])] for p in approx]
        json_folds = [[[round(pt[0]-off, 1), round(pt[1]-off, 1)] for pt in line] for line in fold_lines]
        name = os.path.splitext(f_name)[0]
        anno = {"vertices": vertices, "folds": json_folds, "folding": "3fold", "resolution": [3024, 4032]}
        cv2.imwrite(os.path.join(out_dir, f"{name}_3fold.jpg"), final_crop)
        with open(os.path.join(out_dir, f"{name}_3fold.json"), 'w', encoding='utf-8') as jf:
            json.dump(anno, jf, indent=2)
        print(f"Processed 3fold (Slanted): {name}")

if __name__ == "__main__":
    main()