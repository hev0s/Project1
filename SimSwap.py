import os
import subprocess
import argparse

def run_simswap(pic_a_path, pic_b_path, output_path="output"):
    """
    pic_a_path = image cible (à modifier)
    pic_b_path = image source (visage à transférer)
    """
    command = [
        "python", "test_one_image.py",
        "--isTrain", "false",
        "--name", "people",
        "--Arc_path", "arcface_model/arcface_checkpoint.tar",
        "--pic_a_path", pic_a_path,
        "--pic_b_path", pic_b_path,
        "--output_path", output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"[✔] SimSwap terminé. Résultat dans {output_path}/")
    except subprocess.CalledProcessError as e:
        print(f"[✘] Erreur SimSwap : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Image cible")
    parser.add_argument("--source", required=True, help="Image source (visage)")
    parser.add_argument("--output", default="output", help="Dossier de sortie")
    args = parser.parse_args()

    run_simswap(args.target, args.source, args.output)
