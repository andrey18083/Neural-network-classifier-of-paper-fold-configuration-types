import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random

def autopad(k, p=None): 
    if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1); self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        x = self.cv1(x); y1 = self.m(x); y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class YOLOv8Classifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = Conv(3, 32, 3, 2)          
        self.stage1 = C2f(32, 64, n=3, shortcut=True) 
        self.down1 = Conv(64, 128, 3, 2)
        self.stage2 = C2f(128, 128, n=6, shortcut=True)
        self.down2 = Conv(128, 256, 3, 2)
        self.stage3 = C2f(256, 256, n=6, shortcut=True)
        self.down3 = Conv(256, 512, 3, 2)
        self.stage4 = C2f(512, 512, n=3, shortcut=True)
        self.sppf = SPPF(512, 512)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.stage1(self.stem(x)); x = self.stage2(self.down1(x))
        x = self.stage3(self.down2(x)); x = self.sppf(self.stage4(self.down3(x)))
        return self.classifier(x)

def get_random_image(directory):
    if not os.path.exists(directory):
        print(f"Ошибка: Папка {directory} не существует.")
        return None
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return None
    return os.path.join(directory, random.choice(files))

TEST_DIR = r"C:\hse\3kursovaya\test"
FOLD_LABELS = ["2fold", "3fold", "4fold", "8fold"]
IMG_SIZE = (512, 384)
WEIGHTS_PATH = "yolo_fold_best_95.pth"

def benchmark_inference(image_path):
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Ошибка: Файл весов {WEIGHTS_PATH} не найден!")
        return

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_image = Image.open(image_path).convert("RGB")
    raw_image = raw_image.transpose(Image.ROTATE_270) 
    input_tensor = transform(raw_image).unsqueeze(0)

    results = {}

    for device_name in ['cpu', 'cuda']:
        if device_name == 'cuda' and not torch.cuda.is_available():
            continue
        
        device = torch.device(device_name)
        model = YOLOv8Classifier().to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
        model.eval()

        tensor = input_tensor.to(device)

        with torch.no_grad():
            for _ in range(5):
                _ = model(tensor)

        times = []
        with torch.no_grad():
            for _ in range(30):
                if device_name == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                output = model(tensor)
                
                if device_name == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)

        prob = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(prob).item()
        
        results[device_name] = {
            'time': np.mean(times) * 1000,
            'label': FOLD_LABELS[pred_idx],
            'conf': prob[0][pred_idx].item()
        }

    print(f"Инференс{os.path.basename(image_path)}")
    for dev, res in results.items():
        print(f"Device: {dev.upper()}")
        print(f"  Предсказание: {res['label']} ({res['conf']:.2%})")
        print(f"  Среднее время: {res['time']:.2f} мс")

if __name__ == "__main__":
    random_photo = get_random_image(TEST_DIR)
    benchmark_inference(random_photo)