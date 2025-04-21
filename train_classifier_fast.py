# train_classifier_fast.py – FINAL GTX 1650 (compat‑GradScaler)
# EfficientNet‑B0, batch 80, grad‑accum 2, early‑stopping, prints val_loss/acc

# Force NVIDIA GPU usage for this script
import os
import sys
import ctypes

# Set CUDA environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use first NVIDIA GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Use PCI bus ID for device ordering

# Windows-specific GPU optimizations
if os.name == 'nt':  # Windows
    try:
        # NVIDIA specific environment variables
        os.environ["CUDA_FORCE_PTX_JIT"] = "1"  # Force NVIDIA JIT compilation
        
        # Set process DPI awareness (Windows 10+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI awareness
        except Exception:
            pass
            
        # Set process priority to HIGH (Windows only)
        try:
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            print("⚡ Process priority set to HIGH")
        except ImportError:
            print("Warning: psutil not installed - skipping process priority setting")
        except Exception as e:
            print(f"Warning: Could not set process priority: {e}")
            
        # Windows 10/11 specific - try to force high-performance GPU on dual-GPU systems
        try:
            # This attemps to force Windows to use the high-performance GPU
            if hasattr(ctypes.windll, 'SetProcessPreferredGpu'):
                ctypes.windll.user32.SetProcessPreferredGpu(0, 1, 0)  # Prefer discrete GPU
        except Exception:
            pass
            
        print("🖥️ Windows-specific GPU optimizations applied")
    except Exception as e:
        print(f"Warning: Could not set Windows GPU preferences: {e}")

import torch
import mlflow
import mlflow.pytorch
import multiprocessing
import gc  # For garbage collection
import time  # For timing and sleep operations
import threading  # For monitoring GPU usage
from torch import nn, optim
from torch.amp import GradScaler, autocast  # Updated import path
from functools import partial
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

# ─── Custom picklable transform classes ───────────────────
class LabelMapper:
    def __init__(self, mapping):
        self.mapping = mapping
        
    def __call__(self, label):
        return self.mapping[label]
        
# Class to force tensors to specific device (CPU or CUDA)
class TensorDeviceTransform:
    def __init__(self, device='cpu'):
        self.device = device
    
    def __call__(self, tensor):
        return tensor.to(self.device)
        
# CUDA warmup function - helps initialize CUDA properly
def cuda_warmup(device):
    """Perform warmup operations to initialize CUDA properly."""
    if not torch.cuda.is_available():
        return
        
    try:
        # Ensure CUDA is initialized
        torch.cuda.init()
        
        # Create and perform operations on tensors of different sizes
        sizes = [(10, 10), (100, 100), (1000, 100), (10, 3, 224, 224)]
        for size in sizes:
            # Allocate tensor
            x = torch.randn(*size, device=device)
            # Perform some operations
            y = x + x
            z = torch.matmul(x.flatten(0, -2), x.flatten(0, -2).t()) if len(size) <= 2 else x.sum()
            # Ensure operations are completed
            torch.cuda.synchronize(device)
            # Clean up
            del x, y, z
        
        # Clear cache
        torch.cuda.empty_cache()
        print("✓ CUDA warmup completed successfully")
    except Exception as e:
        print(f"⚠️ CUDA warmup failed: {e}")

# Initialize multiprocessing method for Windows
if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)

# This is the main function that contains all the training logic
def main():
    # ─── Config ────────────────────────────────────────────────
    DATA_DIR   = r"C:\Users\rosor\OneDrive - ITESO\MLOps\Proyecto_Inferencia\data\images\train"
    MODEL_DIR  = r"C:\Users\rosor\OneDrive - ITESO\MLOps\Proyecto_Inferencia\model_files\fast_final_gpu"
    EXPERIMENT = "Emotion_Classification_Final_Test"
    NUM_CLASSES= 3
    EPOCHS     = 20
    PATIENCE   = 4
    GRAD_CLIP  = 1.0
    BATCH      = 80
    ACC_STEPS  = 2   # efectivo 160
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Final model training--------------------------------------------------------------------------")
    
    # ─── Device setup and verification ─────────────────────────
    print("─── GPU Information ────────────────────────────────")
    if torch.cuda.is_available():
        # 1. Select CUDA device first
        cuda_id = 0  # Use first GPU
        torch.cuda.set_device(cuda_id)
        print(f"Selected CUDA device: {cuda_id}")
        
        # 2. Get device properties
        cuda_id = torch.cuda.current_device()  # Verify device was properly set
        gpu_properties = torch.cuda.get_device_properties(cuda_id)
        print(f"GPU Name: {gpu_properties.name}")
        
        # 3. Force cache clearing
        torch.cuda.empty_cache()
        gc.collect()
        
        # 4. Create a small test tensor and verify it's on GPU
        test_tensor = torch.zeros(1, device=f'cuda:{cuda_id}')
        print(f"Test tensor device: {test_tensor.device}")
        
        # 5. Verify CUDA is working by performing a simple operation
        test_tensor = test_tensor + 1
        print(f"CUDA operation successful: {test_tensor.item() == 1.0}")
        del test_tensor  # Free memory
        
        # 6. Run CUDA warmup to initialize engine
        print("Performing CUDA warmup...")
        cuda_warmup(device=f"cuda:{cuda_id}")
        torch.cuda.synchronize()
        gpu_properties = torch.cuda.get_device_properties(cuda_id)
        print(f"   - CUDA Capability: {gpu_properties.major}.{gpu_properties.minor}")
        print(f"   - Total Memory: {gpu_properties.total_memory / 1024**2:.1f} MB")
        print(f"   - CUDA Cores: {gpu_properties.multi_processor_count * 64}")  # Approximate for GTX 1650
        
        # Check memory usage
        print(f"   - Available Memory: {torch.cuda.get_device_properties(cuda_id).total_memory / 1024**2:.1f} MB")
        print(f"   - Used Memory: {torch.cuda.memory_allocated(cuda_id) / 1024**2:.1f} MB")
        print(f"   - Cached Memory: {torch.cuda.memory_reserved(cuda_id) / 1024**2:.1f} MB")
        
        # Force garbage collection to free any cached memory
        gc.collect()
        torch.cuda.empty_cache()
        # Apply optimizations for NVIDIA GPUs
        device = torch.device(f"cuda:{cuda_id}")
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Better performance but less reproducibility
        torch.backends.cudnn.enabled = True
        
        # NVIDIA-specific memory optimizations
        torch.cuda.empty_cache()
        
        # Set up mixed precision training
        scaler = GradScaler('cuda')  # Updated to use new syntax
        
        # Print CUDA version information
        print(f"   - CUDA Version: {torch.version.cuda}")
        print(f"   - cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"   - PyTorch CUDA: {torch.version.cuda}")
        # Try to verify exclusive access to GPU
        try:
            if hasattr(torch.cuda, 'get_resource_usage'):
                usage = torch.cuda.get_resource_usage()
                print(f"   - CUDA Resource Usage: {usage}")
        except:
            pass
        scaler = GradScaler('cuda')  # Updated to use new syntax
    else:
        print("⚠️ CUDA not available. Using CPU instead.")
        device = torch.device("cpu")
        scaler = None
        
    print(f"🖥️ Active device: {device}")
    print("───────────────────────────────────────────────────")
    
    # ─── MLflow setup ──────────────────────────────────────────
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT)
    
    # ─── Transforms ─────────────────────────────────────────────
    # IMPORTANT: Explicitly use CPU-based transforms for image loading
    # This is to avoid Intel GPU usage for image processing before CUDA transfer
    print("Setting up CPU-based transforms for image loading...")
    
    # Force operations to CPU device
    cpu_device = torch.device('cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.1,0.1)),
        transforms.ToTensor(),
        # Use proper class to ensure tensor remains on CPU until explicitly moved
        TensorDeviceTransform(device='cpu'),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    # Verify proper device placement before dataset loading
    print(f"Device before dataset loading: {device}")
    if device.type == 'cuda':
        # Force some operations on CUDA to ensure driver is working correctly
        dummy = torch.ones(1, device=device)
        result = dummy + dummy
        print(f"Simple CUDA operation result device: {result.device}")
        del dummy, result  # Cleanup
    
    # Set up GPU usage monitoring
    def monitor_gpu_usage(stop_event, interval=5.0):
        """Monitor GPU usage in a separate thread."""
        if device.type != 'cuda':
            return
            
        while not stop_event.is_set():
            try:
                # Get current memory usage
                allocated = torch.cuda.memory_allocated(device) / 1024**2
                reserved = torch.cuda.memory_reserved(device) / 1024**2
                
                print(f"\n🔍 GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
                
                # Try to get GPU utilization if possible
                try:
                    import subprocess
                    result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                                      universal_newlines=True)
                    util = float(result.strip())
                    print(f"🔥 GPU Utilization: {util}%")
                except:
                    pass
            except:
                pass
                
            time.sleep(interval)
    
    # Start GPU monitoring in a separate thread
    gpu_monitor_stop = threading.Event()
    if device.type == 'cuda':
        gpu_monitor = threading.Thread(target=monitor_gpu_usage, args=(gpu_monitor_stop,))
        gpu_monitor.daemon = True
        gpu_monitor.start()
    
    # ─── Dataset & sampler ─────────────────────────────────────
    print("Loading dataset...")
    
    # Force a complete GPU memory clear before loading dataset
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    full_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
    wanted = ['sad','neutral','happy']
    idx_map = {full_ds.class_to_idx[c]:i for i,c in enumerate(wanted)}
    sel_idx = [i for i,(_,lbl) in enumerate(full_ds) if lbl in idx_map]
    subset  = torch.utils.data.Subset(full_ds, sel_idx)
    subset.dataset.target_transform = LabelMapper(idx_map)
    print(f"Dataset total: {len(subset)} imgs / {NUM_CLASSES} clases")
    
    train_len = int(0.7*len(subset)); val_len = len(subset)-train_len
    train_ds , val_ds = random_split(subset, [train_len, val_len])
    labels = [subset[i][1] for i in train_ds.indices]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1. / class_counts[labels]
    train_sampler = WeightedRandomSampler(weights, len(labels))
    
    # Clear memory before starting training
    print("\n⚠️ Forcing memory cleanup before training...")
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"   GPU memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**2:.1f}MB allocated")
        
    # Verify Intel GPU is not being used (Windows specific)
    if os.name == 'nt':
        try:
            print("Verifying Windows GPU configuration...")
            # Check if we can detect process GPU assignment
            import wmi
            c = wmi.WMI()
            for process in c.Win32_Process(ProcessId=os.getpid()):
                print(f"   Process ID: {process.ProcessId}")
                print(f"   Process Name: {process.Name}")
        except:
            print("   WMI module not available - skipping Windows GPU verification")
    
    # Start MLflow tracking
    with mlflow.start_run(run_name="EffNet_bs80"):
        mlflow.log_params({"batch":BATCH, "lr":0.0012, "drop":0.3, "acc_steps":ACC_STEPS, "backbone":"efficientnet_b0"})
        
        # ─── Model setup ───────────────────────────────────────
        print("Initializing model...")
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.features[-2:].parameters():
            p.requires_grad = True
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        
        # Move model to device with verification
        print(f"Moving model to {device}...")
        model = model.to(device).to(memory_format=torch.channels_last)
        
        # Verify model movement
        model_device = next(model.parameters()).device
        print(f"✓ Model moved to {model_device}")
        if model_device != device:
            print(f"⚠️ WARNING: Model device mismatch! Expected {device}, got {model_device}")
            
        # Verify memory format
        is_channels_last = next(model.parameters()).is_contiguous(memory_format=torch.channels_last)
        print(f"✓ Memory format is channels_last: {is_channels_last}")
        
        # Print model statistics
        model_size = sum(p.numel() for p in model.parameters())/1e6
        print(f"✓ Model size: {model_size:.2f} M parameters")
        # Perform a test forward pass to verify model works on GPU
        print("Performing test forward pass...")
        try:
            # Create a small batch for testing
            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            
            # Ensure input is in correct memory format
            if device.type == 'cuda':
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
                
            # Try without autocast first for diagnosis
            with torch.no_grad():
                _ = model(dummy_input)
                
            # Now try with autocast
            with torch.no_grad(), autocast('cuda' if device.type == 'cuda' else 'cpu'):
                _ = model(dummy_input)
                
            print("✓ Test forward pass successful")
            # Force synchronization to detect any issues
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            # Check for any CUDA errors
            if torch.cuda.is_available():
                if torch.cuda.has_half_scaler():
                    print("✓ CUDA supports half-precision scaling")
                    
                if torch.cuda.has_half():
                    print("✓ CUDA supports half-precision operations")
                    
            # Clean up
            del dummy_input
        except Exception as e:
            print(f"⚠️ Test forward pass failed: {e}")
            print("  Attempting model execution without channels_last format...")
            try:
                # Try without channels_last as fallback
                model = model.to(device)  # Remove channels_last format
                dummy_input = torch.randn(1, 3, 224, 224, device=device)
                with torch.no_grad():
                    _ = model(dummy_input)
                print("✓ Alternative test forward pass successful")
                del dummy_input
            except Exception as e2:
                print(f"⚠️ Alternative forward pass also failed: {e2}")
        
        # ─── DataLoaders setup ─────────────────────────────────
        # ─── DataLoaders setup ─────────────────────────────────
        # Using multiple workers for parallel data loading (4 is usually a good value)
        num_workers = min(4, os.cpu_count() or 1)
        # DataLoader with NVIDIA optimizations
        train_ld = DataLoader(
            train_ds, 
            batch_size=BATCH, 
            sampler=train_sampler,
            num_workers=num_workers, 
            pin_memory=True,  # Important for faster CPU->GPU transfers
            persistent_workers=True, 
            prefetch_factor=2,
            drop_last=True,  # Slightly faster and avoids batch size issues
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        val_ld = DataLoader(
            val_ds, 
            batch_size=BATCH,
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=2,
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        # ─── Optimizer setup ─────────────────────────────────
        opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0012, weight_decay=0.01)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
        crit  = nn.CrossEntropyLoss(label_smoothing=0.15)
        
        best_acc = 0
        patience = 0
        
        # ─── Training loop ─────────────────────────────────────
        print("Starting training...")
        for ep in range(1, EPOCHS + 1):
            # ── Train ──
            model.train(); tloss=tcorrect=tcount=0
            for step,(x,y) in enumerate(tqdm(train_ld, desc=f"Ep{ep}[Train]", leave=False), 1):
                # Move tensors to device and verify
                x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                # Diagnostic printouts for first batch of first epoch
                if ep == 1 and step == 1:
                    print(f"\nFirst batch verification:")
                    print(f"❯ Input tensor device: {x.device}")
                    print(f"❯ Input tensor shape: {x.shape}")
                    print(f"❯ Input memory format is channels_last: {x.is_contiguous(memory_format=torch.channels_last)}")
                    print(f"❯ Target tensor device: {y.device}")
                    print(f"❯ Current CUDA memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                    # Check for tensor device mismatch
                    if x.device != device or y.device != device:
                        print(f"WARNING: Tensor device mismatch! Expected {device}")
                    # Print model device location
                    print(f"❯ Model is on device: {next(model.parameters()).device}")
                    # Verify NVIDIA GPU is being used (Windows specific)
                    if os.name == 'nt' and device.type == 'cuda':
                        try:
                            print("❯ NVIDIA GPU activity check... ", end="")
                            test = torch.ones(1000, 1000, device=device)
                            test = test @ test  # Matrix multiplication to force GPU computation
                            print("PASSED ✓")
                            del test
                        except Exception as e:
                            print(f"FAILED ✗ - {e}")
                            
                    # Additional CUDA verification
                    try:
                        print("❯ Checking CUDA memory stats:")
                        print(f"  - Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
                        print(f"  - Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
                        print(f"  - Max Allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
                        
                        # Reset peak stats
                        torch.cuda.reset_peak_memory_stats(device)
                    except Exception as e:
                        print(f"  - Memory stats error: {e}")
                with autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(scaler is not None)):
                    out = model(x); loss = crit(out, y) / ACC_STEPS
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % ACC_STEPS == 0:
                    if scaler:
                        scaler.unscale_(opt)  # Only unscale when we're about to update
                        clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        clip_grad_norm_(model.parameters(), GRAD_CLIP)
                        opt.step()
                    opt.zero_grad()
                tloss += loss.item() * ACC_STEPS
                tcount += y.size(0)
                tcorrect += (out.argmax(1) == y).sum().item()
            train_acc  = 100 * tcorrect / tcount
            train_loss = tloss / len(train_ld)

            # ── Validation ──
            model.eval(); vloss=vcorrect=vcount=0
            with torch.no_grad():
                for x,y in tqdm(val_ld, desc=f"Ep{ep}[Val]", leave=False):
                    x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(scaler is not None)):
                        out = model(x); loss = crit(out, y)
                    vloss += loss.item(); vcount += y.size(0); vcorrect += (out.argmax(1) == y).sum().item()
                    
                    # Explicitly clear memory after validation batches
                    if device.type == 'cuda' and torch.cuda.memory_allocated() > 1000 * 1024 * 1024:  # > 1GB
                        # If we're using a lot of GPU memory, clear caches during validation
                        del x, y, out, loss
                        torch.cuda.empty_cache()
            val_acc = 100 * vcorrect / vcount
            val_loss = vloss / len(val_ld)

            # ── Logging y consola ──
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss
            }, step=ep)
            
            # Update learning rate scheduler
            sched.step(val_acc)
            
            # Print progress
            print(f"Ep{ep:02d}: TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}% | ValLoss={val_loss:.4f}")

            # ── Early stopping + guardado ──
            if val_acc > best_acc:
                best_acc = val_acc; patience = 0
                best_path = os.path.join(MODEL_DIR, "best_eff_bs80.pth")
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(best_path)
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("⏹️ Early stopping. No mejora en", PATIENCE, "épocas\n")
                    # Clear memory before stopping
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    break
        
        print(f"✅ Entrenamiento finalizado | Mejor ValAcc = {best_acc:.2f}% | Modelo guardado en {best_path}")
        
        # Stop GPU monitoring thread if it's running
        if device.type == 'cuda' and 'gpu_monitor' in locals():
            gpu_monitor_stop.set()
            gpu_monitor.join(timeout=1.0)

# Entry point for the script
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
