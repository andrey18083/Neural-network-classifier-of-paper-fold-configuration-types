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
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from datetime import datetime


CONFIG = {
    'synth_dir': r"/content/drive/MyDrive/dataset_ready",
    'test_dir': r"/content/drive/MyDrive/test",
    'img_size': (512, 384),
    'batch_size': 8,
    'epochs': 100,
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

FOLD_MAP = {"2fold": 0, "3fold": 1, "4fold": 2, "8fold": 3}

class FoldDataset(Dataset):
    def __init__(self, samples, transform, use_mask, is_real=False):
        self.samples = samples
        self.transform = transform
        self.use_mask = use_mask
        self.is_real = is_real

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.is_real:
            image = image.transpose(Image.ROTATE_270)
        image = self.transform(image)

        if self.use_mask:
            mask = self._load_mask(json_path)

            if random.random() < 0.5:
                mask = mask + torch.randn_like(mask) * 0.05

            if random.random() < 0.3:
                mask = TF.gaussian_blur(mask, kernel_size=5, sigma=random.uniform(0.5, 1.5))

            mask = mask.clamp(0, 1)
        else:
            mask = torch.zeros((1, *CONFIG['img_size']))

        return image, mask, label

    def _load_mask(self, json_path):
        with open(json_path) as f:
            data = json.load(f)

        mask = np.zeros(CONFIG['img_size'], np.uint8)
        
        scale_x = CONFIG['img_size'][1] / 3024.0
        scale_y = CONFIG['img_size'][0] / 4032.0

        for line in data.get("folds", []):
            pts = np.array(line, np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            cv2.polylines(mask, [pts.astype(np.int32)], False, 255, thickness=10)

        mask = torch.from_numpy(mask).float() / 255.0
        return mask.unsqueeze(0)


class FoldNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec1 = block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = block(128, 64)

        self.seg_final = nn.Conv2d(64, 1, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear((512 + 1) * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))

        d1 = self.up1(b)
        d1 = torch.cat([d1, s3], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, s1], dim=1)
        d3 = self.dec3(d3)

        mask_logits = self.seg_final(d3)
        mask_probs = torch.sigmoid(mask_logits)

        m_small = F.interpolate(mask_probs, size=(b.size(2), b.size(3)), mode='bilinear', align_corners=False)

        if self.training:
            if random.random() < 0.25:
                m_small = torch.zeros_like(m_small)
            elif random.random() < 0.25:
                m_small = (m_small + torch.randn_like(m_small) * 0.5).clamp(0, 1)

        combined_features = torch.cat([b, m_small], dim=1)

        cls_logits = self.cls_head(combined_features)

        return mask_logits, cls_logits

class CombinedSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(CONFIG['device']))

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    def forward(self, pred, target):
        return 0.3 * self.bce(pred, target) + 0.7 * self.dice_loss(pred, target)

def train_epoch(model, loader, opt, loss_seg, loss_cls, epoch):
    model.train()
    total = 0
    
    w_seg = 0.7 if epoch < 20 else 0.4
    w_cls = 0.3 if epoch < 20 else 0.6

    for imgs, masks, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
        imgs, masks, labels = imgs.to(CONFIG['device']), masks.to(CONFIG['device']), labels.to(CONFIG['device'])
        opt.zero_grad()
        mask_out, cls_out = model(imgs)

        loss = w_seg * loss_seg(mask_out, masks) + w_cls * loss_cls(cls_out, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader, use_mask_metrics=True):
    model.eval()
    preds, gts = [], []
    total_iou = 0 
    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
            mask_out, cls_out = model(imgs)

            if use_mask_metrics:
                pred_mask = torch.sigmoid(mask_out) > 0.5 
                true_mask = masks > 0.5
                intersection = (pred_mask & true_mask).sum(dim=(1,2,3)).float()
                union = (pred_mask | true_mask).sum(dim=(1,2,3)).float()
                
                mask_exists = true_mask.sum(dim=(1,2,3)) > 0
                if mask_exists.any():
                    iou = intersection[mask_exists] / union[mask_exists]
                    total_iou += iou.mean().item()

            preds.extend(torch.argmax(cls_out, 1).cpu().numpy())
            gts.extend(labels.numpy())

    avg_iou = total_iou / len(loader) if use_mask_metrics else 0.0
    f1 = f1_score(gts, preds, average="macro")
    cm = confusion_matrix(gts, preds)
    return f1, cm, avg_iou

def main():
    train_transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    synth_all = []
    for f in os.listdir(CONFIG['synth_dir']):
        if f.endswith(".jpg"):
            for k, v in FOLD_MAP.items():
                if k in f:
                    img = os.path.join(CONFIG['synth_dir'], f)
                    js = img.replace(".jpg", ".json")
                    synth_all.append((img, js, v))

    train_synth, val_synth_samples = train_test_split(synth_all, test_size=0.1, random_state=42)

    real_all = []
    for f in os.listdir(CONFIG['test_dir']):
        if f.endswith(".jpg"):
            img = os.path.join(CONFIG['test_dir'], f)
            js = img + ".json"
            if os.path.exists(js):
                with open(js) as jf:
                    label = FOLD_MAP[json.load(jf)["folding"]]
                real_all.append((img, None, label))

    random.seed(42) 
    sample_size = int(len(real_all) * 0.2)
    real_subset = random.sample(real_all, sample_size) 

    train_loader = DataLoader(FoldDataset(train_synth, train_transform, use_mask=True), batch_size=CONFIG['batch_size'], shuffle=True)
    val_synth_loader = DataLoader(FoldDataset(val_synth_samples, val_transform, use_mask=True), batch_size=CONFIG['batch_size'])
    val_real_loader = DataLoader(FoldDataset(real_subset, val_transform, use_mask=False, is_real=True), batch_size=CONFIG['batch_size'])

    model = FoldNet().to(CONFIG['device'])
    opt = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)

    loss_seg = CombinedSegLoss()
    loss_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_f1_real = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=7, factor=0.5)
    
    history = []
    log_name = f"log_{datetime.now().strftime('%m%d_%H%M')}.csv"
    best_iou = 0
    for e in range(CONFIG['epochs']):
        loss = train_epoch(model, train_loader, opt, loss_seg, loss_cls, e)
        
        f1_syn, _, iou_syn = eval_epoch(model, val_synth_loader, use_mask_metrics=True)
        f1_real, cm_real, _ = eval_epoch(model, val_real_loader, use_mask_metrics=False)
        
        scheduler.step(f1_real)
        
        print(f"\nEpoch {e+1} | Loss: {loss:.4f} | LR: {opt.param_groups[0]['lr']:.6f}")
        print(f"SYNTH -> F1: {f1_syn:.3f} | IoU: {iou_syn:.4f}")
        print(f"REAL  -> F1: {f1_real:.3f}")
        print(cm_real)

        if f1_real > best_f1_real:
            best_f1_real = f1_real
            torch.save(model.state_dict(), "best_fold_model.pth")
            print("--- New Leader Saved ---")

        if iou_syn > best_iou:
            best_iou = iou_syn
            torch.save(model.state_dict(), "best_iou_model.pth")
            print("Best IoU")

        history.append({
            "epoch": e+1, "loss": loss, "iou_synth": iou_syn, 
            "f1_real": f1_real, "lr": opt.param_groups[0]['lr']
        })
        pd.DataFrame(history).to_csv(log_name, index=False)

if __name__ == "__main__":
    main()