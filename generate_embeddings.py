# generate_embeddings.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
import torch
import multiprocessing
from tqdm import tqdm
import sys
import time

# Configure CPU usage (adjust these values as needed)
CPU_THREADS = max(1, multiprocessing.cpu_count() // 2)  # Use half of available CPU cores
NUM_WORKERS = 2  # Number of workers for data loading

# Limit CPU usage
torch.set_num_threads(CPU_THREADS)
# Limit NumPy threads via environment variables
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS) 
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["BLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(CPU_THREADS)

CLASSES = ['happy', 'neutral', 'sad']
INPUT_DIR = r"C:\Users\rosor\OneDrive - ITESO\MLOps\Proyecto_Inferencia\data\images\train"
OUTPUT_CSV = r"C:\Users\rosor\OneDrive - ITESO\MLOps\Proyecto_Inferencia\image_embeddings.csv"
BATCH_SIZE = 64
TARGET_SIZE = (299, 299)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.inception_v3(pretrained=True, aux_logits=True)
model.aux_logits = False
model.fc = torch.nn.Identity()
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def process_images():
    start_time = time.time()
    all_embeddings, all_labels = [], []
    
    # Count total images to process
    total_images = 0
    for class_name in CLASSES:
        class_path = os.path.join(INPUT_DIR, class_name)
        total_images += len(os.listdir(class_path))
    
    print(f"Total de imágenes a procesar: {total_images}")
    print(f"Usando dispositivo: {device}")
    
    processed_images = 0
    errors_count = 0
    
    # Create progress bar for all classes
    class_progress = tqdm(CLASSES, desc="Procesando clases", unit="clase")
    
    for class_name in class_progress:
        class_start_time = time.time()
        class_path = os.path.join(INPUT_DIR, class_name)
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path)]
        
        class_progress.set_description(f"Procesando clase: {class_name}")
        print(f"\nClase: {class_name}, Total: {len(image_files)} imágenes")
        
        # Create progress bar for batches within this class
        batch_progress = tqdm(range(0, len(image_files), BATCH_SIZE), 
                             desc=f"Batches de {class_name}", 
                             unit="batch")
        
        class_success = 0
        class_errors = 0
        
        for i in batch_progress:
            batch_paths = image_files[i:i + BATCH_SIZE]
            batch_tensors = []
            batch_errors = 0
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('L')
                    img = np.stack([img] * 3, axis=-1)
                    img = Image.fromarray(img)
                    img = preprocess(img)
                    batch_tensors.append(img)
                except Exception as e:
                    batch_errors += 1
                    class_errors += 1
                    errors_count += 1
                    error_msg = f"Error en {path}: {e}"
                    print(error_msg)
                    # Write error to log file
                    with open("embedding_errors.log", "a") as f:
                        f.write(f"{error_msg}\n")

            if not batch_tensors:
                batch_progress.set_description(f"Batch {i//BATCH_SIZE+1} sin imágenes válidas")
                continue

            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                embeddings = model(batch).view(batch.size(0), -1).cpu().numpy()

            all_embeddings.extend(embeddings)
            all_labels.extend([class_name] * len(embeddings))
            
            # Update processed count and progress information
            batch_success = len(batch_tensors)
            class_success += batch_success
            processed_images += batch_success
            
            # Update progress bar with current stats
            batch_progress.set_description(
                f"Batch {i//BATCH_SIZE+1}/{len(image_files)//BATCH_SIZE+1} - Éxito: {batch_success}/{len(batch_paths)}"
            )
            
            # Display overall progress
            overall_progress = (processed_images / total_images) * 100
            batch_progress.set_postfix({
                'Total procesado': f"{processed_images}/{total_images}",
                'Progreso total': f"{overall_progress:.1f}%"
            })
            
            # Flush output to ensure real-time updates
            sys.stdout.flush()
        
        class_time = time.time() - class_start_time
        print(f"Clase {class_name} completada en {class_time:.2f} segundos")
        print(f"Imágenes procesadas: {class_success}/{len(image_files)} ({class_errors} errores)")
    
    total_time = time.time() - start_time
    print(f"\nProcesamiento completado en {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"Total procesado: {processed_images}/{total_images} imágenes ({errors_count} errores)")
    
    if all_embeddings:
        df = pd.DataFrame(all_embeddings)
        df['label'] = all_labels
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Embeddings guardados en {OUTPUT_CSV}")
    else:
        print("ERROR: No se generaron embeddings para guardar.")

def print_cpu_info():
    """Print information about CPU usage configuration."""
    print(f"CPU Configuration:")
    print(f"CPU Configuration:")
    print(f"- PyTorch threads: {torch.get_num_threads()}")
    print(f"- NumPy threads (via env vars): {os.environ.get('OMP_NUM_THREADS')}")
    print(f"- CPU threads limit: {CPU_THREADS} (of {multiprocessing.cpu_count()} available)")
    print(f"- Workers: {NUM_WORKERS}")
    print("-" * 50)
if __name__ == "__main__":
    print_cpu_info()
    try:
        print("Iniciando generación de embeddings...")
        process_images()
        print("Proceso finalizado correctamente")
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
