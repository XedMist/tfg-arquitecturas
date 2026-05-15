import subprocess
import sys
import os

def main():
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arch_dir = os.path.join(src_dir, "models", "arch")
    
    # Capture any additional arguments passed to this script
    # This allows passing hydra config overrides directly, e.g., training.epochs=100 optimizer.lr=1e-4
    extra_args = sys.argv[1:]
    
    # Dynamically find all architectures (python files excluding __init__.py and starting with _)
    archs = []
    for file in os.listdir(arch_dir):
        if file.endswith(".py") and not file.startswith("__"):
            arch_name = file[:-3]
            archs.append(arch_name)
    
    archs.sort()
    
    print(f"Found architectures: {archs}")
    if extra_args:
        print(f"Applying general config overrides: {' '.join(extra_args)}")
    
    for arch in archs:
        print(f"\n{'='*50}")
        print(f"Running training for architecture: {arch}")
        print(f"{'='*50}\n")
        
        cmd = [
            sys.executable, "train.py",
            f"model.arch={arch}",
            "data.subset_fraction=1.0",
            f"experiment.name={arch}_whole_imagenette"
        ] + extra_args
        
        try:
            # Run the command, stream output to console
            subprocess.run(cmd, cwd=src_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while training {arch}: {e}")
            continue

if __name__ == "__main__":
    main()
