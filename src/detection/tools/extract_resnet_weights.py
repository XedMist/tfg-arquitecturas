import argparse
import os
from pathlib import Path

import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Extrae los pesos del backbone (model_state_dict) del checkpoint de clasificación.")
    parser.add_argument("checkpoint", type=str, help="Ruta al checkpoint de origen (.ckpt)")
    parser.add_argument("output", type=str, help="Ruta donde se guardarán los pesos extraídos (.pth)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ckpt_path}")
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando checkpoint desde {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "model_state_dict" not in ckpt:
        raise KeyError("El checkpoint no contiene 'model_state_dict'. ¿Es un checkpoint válido del trainer?")
        
    state_dict = ckpt["model_state_dict"]
    
    print(f"Guardando pesos (model_state_dict) en {out_path}...")
    torch.save(state_dict, out_path)
    
    print("¡Extracción completada con éxito!")

if __name__ == "__main__":
    main()
