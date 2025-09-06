# main_cifar10.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# pakai modul kamu
from kcn import ConvNeXtKAN          # (ConvNeXt + KAN head)  [file: kcn.py]
from train_and_test import train, test   # loop latih & evaluasi [file: train_and_test.py]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ======================
# DATA: CIFAR-10
# ======================
# Pilih salah satu root yang ada (ubah kalau perlu)
CANDIDATE_ROOTS = [
    "/kaggle/input/cifar10/cifar10"                        # lokal
]

# 1) PAKAI TORCHVISION.CIFAR10 (paling gampang, auto-download ke ./cifar10)
use_torchvision_cifar = True

# Normalisasi ImageNet untuk backbone ConvNeXt (pretrained)
train_tf = transforms.Compose([
    transforms.Resize(224),                # CIFAR-10 32x32 -> upsample 224 buat ConvNeXt
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

if use_torchvision_cifar:
    # Download otomatis ke ./cifar10
    train_dataset = datasets.CIFAR10(root="./cifar10", train=True,  download=True, transform=train_tf)
    test_dataset  = datasets.CIFAR10(root="./cifar10", train=False, download=True, transform=test_tf)
    class_names = train_dataset.classes  # ['airplane', ..., 'truck']
else:
    # 2) Alternatif: kalau kamu sudah punya folder "cifar10/train" & "cifar10/test" (subfolder per kelas)
    #    isi salah satu path di CANDIDATE_ROOTS di atas.
    import os
    data_root = next((p for p in CANDIDATE_ROOTS if os.path.isdir(p)), None)
    assert data_root is not None, "Folder cifar10 tidak ditemukan. Set use_torchvision_cifar=True atau periksa path."
    from torchvision import datasets as dsets
    train_dataset = dsets.ImageFolder(root=f"{data_root}/train", transform=train_tf)
    test_dataset  = dsets.ImageFolder(root=f"{data_root}/test",  transform=test_tf)
    class_names = train_dataset.classes

batch_size  = 128
num_workers = 4 if torch.cuda.is_available() else 2
pin_memory  = torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

# ======================
# MODEL, LOSS, OPTIM
# ======================
model = ConvNeXtKAN().to(device)   # dari kcn.py (ConvNeXt pretrained + 2x KANLinear)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
# scheduler opsional:
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20*len(train_loader))

# ======================
# TRAIN & EVAL
# ======================
num_epochs = 20
best_acc = 0.0
best_path = "convnext_kan_cifar10.pth"

for epoch in range(1, num_epochs+1):
    print(f"Epoch {epoch}/{num_epochs}")
    train(model, train_loader, criterion, optimizer, device)      # dari train_and_test.py
    # if 'scheduler' in locals(): scheduler.step()
    test(model, test_loader, criterion, device, class_names)      # dari train_and_test.py
    # Catatan: fungsi test() kamu sudah simpan metrics + confusion matrix png

    # (opsional) simpan best by accuracy — di sini kita ambil dari log test() saja;
    # bisa juga compute ulang akurasi singkat di sini kalau mau strict.

# simpan final
torch.save(model.state_dict(), best_path)
print(f"Model saved as '{best_path}'")
