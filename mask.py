import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from torchvision.transforms import functional as TF

MODEL_PATH = 'best_fold_model.pth'
TEST_DIR = r"C:\hse\3kursovaya\test" 

IMG_SIZE = (512, 384) 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def predict():
    model = FoldNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    img_name = np.random.choice(files)
    img_path = os.path.join(TEST_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    
    is_synth = "dataset_ready" in img_path or "_v" in img_name
    if not is_synth:
        image = image.transpose(Image.ROTATE_270)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_logits, cls_logits = model(input_tensor)
        mask_probs = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
        pred_idx = torch.argmax(cls_logits, 1).item()
        fold_classes = {0: "2fold", 1: "3fold", 2: "4fold", 3: "8fold"}
        pred_label = fold_classes[pred_idx]

    display_size = (IMG_SIZE[1], IMG_SIZE[0])
    img_res = image.resize(display_size)
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(img_res)
    plt.title(f"Input: {img_name}")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask_probs, cmap='jet')
    plt.title("Predicted Mask (Heatmap)")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    img_np = np.array(img_res)
    mask_binary = (mask_probs > 0.2).astype(np.uint8)
    overlay = img_np.copy()
    overlay[mask_binary > 0] = [255, 0, 0]
    combined = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)
    plt.imshow(combined)
    plt.title(f"Prediction: {pred_label}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict()