@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo NVIDIA GPU Training Launcher
echo ===================================================
echo.

:: Set NVIDIA environment variables
echo Setting NVIDIA environment variables...
set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_FORCE_PTX_JIT=1

:: Force high-performance NVIDIA GPU for notebook GPUs
set __COMPAT_LAYER=RunAsInvoker
set __NV_PRIME_RENDER_OFFLOAD=1
set __GLX_VENDOR_LIBRARY_NAME=nvidia
set __VK_LAYER_NV_optimus=NVIDIA_only

:: These are Windows-specific to force GPU selection
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.5,max_split_size_mb:128
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_USE_CUDNN=1

:: Make sure Python can access the environment variables
echo Making environment variables visible to Python...
setx CUDA_VISIBLE_DEVICES 0 /M >nul 2>&1
setx CUDA_DEVICE_ORDER PCI_BUS_ID /M >nul 2>&1
setx CUDA_FORCE_PTX_JIT 1 /M >nul 2>&1

:: Verify NVIDIA GPU is available
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: NVIDIA GPU not found or driver not working.
    echo Please ensure your NVIDIA drivers are installed correctly.
    goto :error
)

:: Kill any existing Python processes (optional - uncomment if needed)
:: echo Cleaning up existing Python processes...
:: taskkill /F /IM python.exe >nul 2>&1
:: taskkill /F /IM pythonw.exe >nul 2>&1

:: Clear GPU memory
echo Cleaning GPU memory...
nvidia-smi --gpu-reset >nul 2>&1

:: Print diagnostics
echo.
echo GPU diagnostics:
nvidia-smi
echo.

:: Get Python executable path
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH.
    goto :error
)

echo Starting training with NVIDIA GPU...
echo.
echo ===================================================
echo.

:: Run the Python script with high priority
start /B /HIGH python train_classifier_fast.py

if %errorlevel% neq 0 (
    echo.
    echo Error running training script. Check the logs for details.
    goto :error
)

echo.
echo Training started successfully. See output for progress.
goto :end

:error
echo.
echo ===================================================
echo ERROR: Training could not be started properly.
echo ===================================================
pause
exit /b 1

:end
echo.
echo ===================================================
echo Training process launched successfully.
echo ===================================================
endlocal

