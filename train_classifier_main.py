# train_classifier_fast.py – GTX 1650 (batch 80, grad‑accum 2, early‑stopping)
# EfficientNet‑B0, mixed‑precision, channels‑last, sampler balanceado
# Imprime por época: TrainAcc, ValAcc, ValLoss
# Guarda mejor modelo en model_files/fast_v2/best_eff_bs80.pth

import os, torch, mlflow, mlflow.pytorch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# ─── Configuración ──────────────────────────────────────────
DATA_DIR   = r"C:\Users\rosor\OneDrive - ITESO\MLOps\Proyecto_Inferencia\data\images\train"
MODEL_DIR  = r"C:\Users\rosor\OneDrive - ITESO\MLOps\Proyecto_Inferencia\model_files\fast_v2"
EXPERIMENT = "Emotion_Fast_GPU"

NUM_CLASSES = 3
EPOCHS      = 20
PATIENCE    = 4
GRAD_CLIP   = 1.0
BATCH       = 80       # VRAM ~3 GB en GTX 1650
ACC_STEPS   = 2        # acumulación → batch efectivo 160

os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Device:", device)

# mixed‑precision scaler
scaler = GradScaler() if device.type == "cuda" else None
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ─── Transformaciones ──────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=12, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Dataset y sampler balanceado ──────────────────────────
full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
wanted  = ['sad', 'neutral', 'happy']
idx_map = {full_ds.class_to_idx[c]: i for i, c in enumerate(wanted)}

subset_idx = [i for i, (_, lbl) in enumerate(full_ds) if lbl in idx_map]
subset     = torch.utils.data.Subset(full_ds, subset_idx)
subset.dataset.target_transform = lambda x: idx_map[x]

print(f"Dataset: {len(subset)} imágenes / {NUM_CLASSES} clases")

train_len = int(0.7 * len(subset))
val_len   = len(subset) - train_len
train_ds, val_ds = random_split(subset, [train_len, val_len])

labels      = [subset[i][1] for i in train_ds.indices]
class_count = torch.bincount(torch.tensor(labels))
weights     = 1. / class_count[labels]
train_sampler = WeightedRandomSampler(weights, len(labels))

# ─── MLflow ────────────────────────────────────────────────
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="EffNet_bs80"):
    mlflow.log_params({
        "batch": BATCH, "lr": 0.0012, "drop": 0.3,
        "acc_steps": ACC_STEPS, "backbone": "efficientnet_b0"
    })

    # EfficientNet‑B0
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[-2:].parameters():
        p.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.to(device).to(memory_format=torch.channels_last)

    # DataLoaders (num_workers = 0 → sin multiprocessing en Windows)
    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=BATCH,
        num_workers=0,
        pin_memory=True
    )

    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=0.0012, weight_decay=0.01)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max',
                                                 factor=0.5, patience=2)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.15)

    best_acc = 0
    patience  = 0

    for ep in range(1, EPOCHS + 1):
        # ── Entrenamiento ──
        model.train()
        tloss = tcorrect = tcount = 0

        for step, (x, y) in enumerate(tqdm(train_ld, desc=f"Ep{ep}[Train]", leave=False), 1):
            x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=(scaler is not None)):
                out  = model(x)
                loss = crit(out, y) / ACC_STEPS

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()

            if step % ACC_STEPS == 0:
                clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()

            tloss    += loss.item() * ACC_STEPS
            tcount   += y.size(0)
            tcorrect += (out.argmax(1) == y).sum().item()

        train_acc  = 100 * tcorrect / tcount
        train_loss = tloss / len(train_ld)

        # ── Validación ──
        model.eval()
        vloss = vcorrect = vcount = 0
        with torch.no_grad():
            for x, y in tqdm(val_ld, desc=f"Ep{ep}[Val]", leave=False):
                x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast(enabled=(scaler is not None)):
                    out  = model(x)
                    loss = crit(out, y)

                vloss    += loss.item()
                vcount   += y.size(0)
                vcorrect += (out.argmax(1) == y).sum().item()

        val_acc  = 100 * vcorrect / vcount
        val_loss = vloss / len(val_ld)

        # ── Logging y consola ──
        mlflow.log_metrics({
            "train_acc": train_acc, "train_loss": train_loss,
            "val_acc": val_acc,     "val_loss": val_loss
        }, step=ep)
        sched.step(val_acc)

        print(f"Ep{ep:02d}: "
              f"TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}% | ValLoss={val_loss:.4f}")

        # ── Early stopping ──
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            best_path = os.path.join(MODEL_DIR, "best_eff_bs80.pth")
            torch.save(model.state_dict(), best_path)
            mlflow.log_artifact(best_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"⏹️ Early stopping (paciencia {PATIENCE})\n")
                break

    print(f"✅ Entrenamiento completo | Mejor ValAcc = {best_acc:.2f}%\n"
          f"   Modelo guardado en {best_path}")
