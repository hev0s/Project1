import os
import subprocess
import argparse

def run_facefusion(source_path, target_path, output_path="fusion_output.jpg"):
    """
    Appelle FaceFusion pour un swap entre deux images
    """
    command = [
        "python", "run.py",
        "--source-image", source_path,
        "--target-media", target_path,
        "--output-path", output_path,
        "--execution-providers", "cpu"  # ou "cuda" si GPU
    ]

    try:
        subprocess.run(command, check=True)
        print(f"[✔] FaceFusion terminé : {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[✘] Erreur FaceFusion : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default="fusion_output.jpg")
    args = parser.parse_args()

    run_facefusion(args.source, args.target, args.output)
