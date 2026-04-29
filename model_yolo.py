import os
import json
import cv2
import numpy as np
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CONFIG = {
    'synth_dir': r"//kaggle/input/datasets/andrey18083/fold-class/dataset_ready3/dataset_ready",
    'test_dir': r"/kaggle/input/datasets/andrey18083/fold-class/test/test",
    'img_size': (512, 384),
    'batch_size': 16,
    'epochs': 100,
    'lr': 3e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

FOLD_MAP = {"2fold": 0, "3fold": 1, "4fold": 2, "8fold": 3}

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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down1(x))
        x = self.stage3(self.down2(x))
        x = self.sppf(self.stage4(self.down3(x)))
        return self.classifier(x)   

class YOLODataSet(Dataset):
    def __init__(self, samples, transform, is_test=False):
        self.samples = samples
        self.transform = transform
        self.is_test = is_test

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_p, label = self.samples[idx]
        image = Image.open(img_p).convert("RGB")
        if self.is_test:
            image = image.transpose(Image.ROTATE_270)
        return self.transform(image), label

def main():
    train_transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.RandomResizedCrop(CONFIG['img_size'], scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomInvert(p=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    synth_samples = []
    for f in os.listdir(CONFIG['synth_dir']):
        if f.endswith(".jpg"):
            for k, v in FOLD_MAP.items():
                if k in f: synth_samples.append((os.path.join(CONFIG['synth_dir'], f), v))
    
    real_test_all = []
    for f in os.listdir(CONFIG['test_dir']):
        if f.lower().endswith(".jpg"):
            img_p = os.path.join(CONFIG['test_dir'], f)
            js_p = img_p + ".json"
            if os.path.exists(js_p):
                with open(js_p) as jf:
                    label = FOLD_MAP[json.load(jf)["folding"]]
                real_test_all.append((img_p, label))

    _, fast_test_samples = train_test_split(real_test_all, test_size=0.2, random_state=42, stratify=[x[1] for x in real_test_all])
    
    print(f"Dataset ready. Train (Synth): {len(synth_samples)}, Test (Real 20%): {len(fast_test_samples)}")

    train_loader = DataLoader(YOLODataSet(synth_samples, train_transform), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(YOLODataSet(fast_test_samples, val_transform, is_test=True), batch_size=CONFIG['batch_size'])

    model = YOLOv8Classifier().to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_f1 = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        t_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            t_loss += loss.item()
        
        scheduler.step()

        model.eval()
        all_p, all_g = [], []
        with torch.no_grad():
            for i, l in val_loader:
                out = model(i.to(CONFIG['device']))
                all_p.extend(out.argmax(1).cpu().numpy())
                all_g.extend(l.numpy())
        
        f1 = f1_score(all_g, all_p, average='macro')
        print(f"Epoch {epoch+1} | Loss: {t_loss/len(train_loader):.4f} | F1: {f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "yolo_fold_best.pth")
            print('Сохранили')
            print(confusion_matrix(all_g, all_p))

if __name__ == "__main__":
    main()