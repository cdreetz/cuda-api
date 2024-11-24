from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import subprocess
import os
import time
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import re

app = FastAPI(title="CUDA Development API")

# Base directory for all operations
BASE_DIR = Path("/tmp/cuda_workspace")
BASE_DIR.mkdir(exist_ok=True)

class WriteFileRequest(BaseModel):
    filename: str
    content: str
    description: Optional[str] = "No description provided"

class CompileRequest(BaseModel):
    filename: str
    options: Dict[str, Any] = Field(
        default_factory=lambda: {
            "optimization": "O3",
            "gpu_arch": "sm_70",  # Default arch, should be configurable
            "include_dirs": [],
            "extra_flags": []
        }
    )

class BenchmarkConfig(BaseModel):
    executable: str
    args: List[str] = []
    warmup_runs: int = 5
    timing_runs: int = 100
    flops_calculation: Optional[Dict[str, Any]] = None  # For calculating FLOPS if needed

def validate_cuda_filename(filename: str) -> bool:
    """Validate CUDA filename and check for path traversal."""
    if not re.match(r'^[a-zA-Z0-9_-]+\.(cu|cuh)$', filename):
        return False
    file_path = BASE_DIR / filename
    return file_path.resolve().is_relative_to(BASE_DIR)

@app.post("/write")
async def write_file(request: WriteFileRequest):
    """Write a CUDA source file."""
    if not validate_cuda_filename(request.filename):
        raise HTTPException(
            status_code=400, 
            detail="Invalid filename. Use only letters, numbers, underscores, and hyphens with .cu or .cuh extension"
        )
    
    try:
        file_path = BASE_DIR / request.filename
        with open(file_path, "w") as f:
            f.write(request.content)
        
        return {
            "status": "success",
            "message": f"File written successfully",
            "file_info": {
                "path": str(file_path),
                "size": os.path.getsize(file_path),
                "description": request.description
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """List all CUDA source files in the workspace."""
    try:
        files = []
        for ext in [".cu", ".cuh"]:
            files.extend(BASE_DIR.glob(f"*{ext}"))
        
        return {
            "files": [
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime
                }
                for f in files
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file/{filename}")
async def read_file(filename: str):
    """Read a CUDA source file."""
    if not validate_cuda_filename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    try:
        file_path = BASE_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file_path, "r") as f:
            content = f.read()
        
        return {
            "filename": filename,
            "content": content,
            "size": os.path.getsize(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compile")
async def compile_code(request: CompileRequest):
    """Compile a CUDA source file."""
    if not validate_cuda_filename(request.filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    try:
        source_path = BASE_DIR / request.filename
        output_path = BASE_DIR / f"{source_path.stem}.out"
        
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source file not found")
        
        # Build command with user-specified options
        cmd = ["nvcc", str(source_path), "-o", str(output_path)]
        
        # Add optimization level
        if "optimization" in request.options:
            cmd.append(f"-{request.options['optimization']}")
        
        # Add GPU architecture
        if "gpu_arch" in request.options:
            cmd.append(f"-arch={request.options['gpu_arch']}")
        
        # Add include directories
        for inc_dir in request.options.get("include_dirs", []):
            cmd.extend(["-I", inc_dir])
        
        # Add any extra flags
        cmd.extend(request.options.get("extra_flags", []))
        
        # Run compilation
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            return {
                "status": "error",
                "command": " ".join(cmd),
                "error": process.stderr
            }
        
        return {
            "status": "success",
            "command": " ".join(cmd),
            "output_file": str(output_path),
            "compiler_output": process.stdout
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark")
async def benchmark(config: BenchmarkConfig):
    """Run a benchmark on a compiled CUDA executable."""
    try:
        executable_path = BASE_DIR / config.executable
        if not executable_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Executable {config.executable} not found"
            )
        
        # Ensure executable has run permission
        executable_path.chmod(0o755)
        
        results = {
            "warmup_runs": [],
            "timing_runs": [],
            "summary": {}
        }
        
        cmd = [str(executable_path)] + config.args
        
        # Warmup runs
        print(f"Performing {config.warmup_runs} warmup runs...")
        for _ in range(config.warmup_runs):
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                return {
                    "status": "error",
                    "message": "Benchmark failed during warmup",
                    "error": process.stderr
                }
            results["warmup_runs"].append(process.stdout)
        
        # Timing runs
        print(f"Performing {config.timing_runs} timing runs...")
        timing_data = []
        
        for i in range(config.timing_runs):
            start_time = time.perf_counter()
            process = subprocess.run(cmd, capture_output=True, text=True)
            end_time = time.perf_counter()
            
            if process.returncode != 0:
                return {
                    "status": "error",
                    "message": f"Benchmark failed at iteration {i}",
                    "error": process.stderr
                }
            
            elapsed_time = end_time - start_time
            timing_data.append(elapsed_time)
            
            # Try to parse JSON output if present
            try:
                run_data = json.loads(process.stdout)
                results["timing_runs"].append(run_data)
            except json.JSONDecodeError:
                results["timing_runs"].append({
                    "raw_output": process.stdout,
                    "elapsed_time": elapsed_time
                })
        
        # Calculate summary statistics
        import numpy as np
        timing_array = np.array(timing_data)
        
        results["summary"] = {
            "mean_time": float(np.mean(timing_array)),
            "std_dev": float(np.std(timing_array)),
            "min_time": float(np.min(timing_array)),
            "max_time": float(np.max(timing_array)),
            "median_time": float(np.median(timing_array))
        }
        
        # Calculate FLOPS if configuration provided
        if config.flops_calculation:
            # Example for matrix multiplication
            # flops = 2 * M * N * K for gemm
            if "operation" in config.flops_calculation:
                if config.flops_calculation["operation"] == "gemm":
                    M = config.flops_calculation.get("M", 1024)
                    N = config.flops_calculation.get("N", 1024)
                    K = config.flops_calculation.get("K", 1024)
                    flops = 2 * M * N * K
                    results["summary"]["gflops"] = (flops / 1e9) / results["summary"]["mean_time"]
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_local_ip():
    """Get the local network IP address."""
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
    print(f"🚀 CUDA Development API Server Starting")
    print("="*50)
    print(f"📡 Local Network Access:")
    print(f"   http://{local_ip}:{port}")
    print("\n💡 Example workflow:")
    print(f"1. Write a CUDA kernel:")
    print(f"   curl -X POST \"http://{local_ip}:{port}/write\" \\")
    print("        -H \"Content-Type: application/json\" \\")
    print('        -d \'{"filename": "kernel.cu", "content": "#include <cuda_runtime.h>..."}\'')
    print(f"\n2. Compile the kernel:")
    print(f"   curl -X POST \"http://{local_ip}:{port}/compile\" \\")
    print("        -H \"Content-Type: application/json\" \\")
    print('        -d \'{"filename": "kernel.cu", "options": {"optimization": "O3"}}\'')
    print(f"\n3. Run benchmark:")
    print(f"   curl -X POST \"http://{local_ip}:{port}/benchmark\" \\")
    print("        -H \"Content-Type: application/json\" \\")
    print('        -d \'{"executable": "kernel.out", "timing_runs": 100}\'')
    print("\n📝 API Documentation:")
    print(f"   http://{local_ip}:{port}/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
