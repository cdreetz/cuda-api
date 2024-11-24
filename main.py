from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import subprocess
import os
import time
import json
from typing import Optional, Dict, Any
from pathlib import Path

app = FastAPI(title="CUDA Operations API")

# Base directory for all operations
BASE_DIR = Path("/tmp/cuda_workspace")
BASE_DIR.mkdir(exist_ok=True)

class FileWriteRequest(BaseModel):
    filename: str
    content: str
    kernel_name: Optional[str] = None

class CompileRequest(BaseModel):
    filename: str
    options: Optional[Dict[str, Any]] = None

class BenchmarkRequest(BaseModel):
    kernel_name: str
    input_size: int
    num_iterations: int = 1000

@app.post("/write")
async def write_file(request: FileWriteRequest):
    """Write CUDA source code to a file."""
    try:
        file_path = BASE_DIR / request.filename
        
        # Ensure we're not writing outside BASE_DIR
        if not file_path.resolve().is_relative_to(BASE_DIR):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        with open(file_path, "w") as f:
            f.write(request.content)
            
        return {
            "status": "success",
            "message": f"File written to {request.filename}",
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compile")
async def compile_code(request: CompileRequest):
    """Compile CUDA source code."""
    try:
        source_path = BASE_DIR / request.filename
        output_path = source_path.with_suffix(".out")
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source file not found")
        
        # Build nvcc command with default options
        cmd = [
            "nvcc",
            str(source_path),
            "-o", str(output_path),
            "-O3"  # Default optimization level
        ]
        
        # Add any additional compilation options
        if request.options:
            for key, value in request.options.items():
                cmd.extend([f"-{key}", str(value)])
        
        # Run compilation
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            return {
                "status": "error",
                "message": "Compilation failed",
                "error": process.stderr
            }
            
        return {
            "status": "success",
            "message": "Compilation successful",
            "output_path": str(output_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark")
async def benchmark_kernel(request: BenchmarkRequest):
    """Benchmark a CUDA kernel."""
    try:
        results = []
        
        # Run the benchmark multiple times
        for i in range(request.num_iterations):
            # You'll need to implement your own benchmarking logic here
            # This is just a placeholder that assumes your compiled program
            # accepts certain arguments
            
            executable = BASE_DIR / f"{request.kernel_name}.out"
            if not executable.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Executable {request.kernel_name}.out not found"
                )
            
            start_time = time.perf_counter()
            
            process = subprocess.run(
                [str(executable), str(request.input_size)],
                capture_output=True,
                text=True
            )
            
            end_time = time.perf_counter()
            
            if process.returncode != 0:
                return {
                    "status": "error",
                    "message": "Benchmark failed",
                    "error": process.stderr
                }
            
            results.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = sum(results) / len(results)
        min_time = min(results)
        max_time = max(results)
        
        return {
            "status": "success",
            "statistics": {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "num_iterations": request.num_iterations
            },
            "kernel_name": request.kernel_name,
            "input_size": request.input_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional utility endpoints

@app.get("/health")
async def health_check():
    """Check if the API is running and CUDA is available."""
    try:
        # Check CUDA availability
        process = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True
        )
        
        cuda_available = process.returncode == 0
        
        return {
            "status": "healthy",
            "cuda_available": cuda_available,
            "workspace_path": str(BASE_DIR)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn

    port = 8000
    local_ip = get_local_ip()

    print("\n" + "="*50)
    print(f" http://{local_ip}:{port}")
    print("="*50)
    print("API Documentation:")
    print(f" http://{local_ip}:{port}/docs")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=port)
