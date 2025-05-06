import os
import cv2
import numpy as np
import random
from tqdm import tqdm


def add_corrosion(img, texture_path, alpha=0.3):
    """Overlay a rust texture onto the image."""
    rust = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    rust = cv2.resize(rust, (img.shape[1], img.shape[0]))
    # If texture has alpha channel, split and composite
    if rust.shape[2] == 4:
        b, g, r, a = cv2.split(rust)
        rust_rgb = cv2.merge([b, g, r])
        mask = cv2.merge([a, a, a]) / 255.0
        blended = img * (1 - mask * alpha) + rust_rgb * (mask * alpha)
        return blended.astype(np.uint8)
    else:
        return cv2.addWeighted(img, 1 - alpha, rust, alpha, 0)


def add_leak(img, hue_shift=20, blur_ksize=(15, 15)):
    """Simulate a bluish liquid spill + blur."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Shift hue towards blue
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_shift) % 180
    tinted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.GaussianBlur(tinted, blur_ksize, 0)


def simulate(input_dir, output_root, texture_path, frac_corrosion=0.5):
    """
    Generate a 3-class synthetic dataset:
      - cracked: original SDNET cracked images (symlinked)
      - corrosion: non-cracked images with rust overlay
      - leak: non-cracked images with blue-tint blur
    """
    # Prepare output directories
    os.makedirs(output_root, exist_ok=True)
    for cls in ["cracked", "corrosion", "leak"]:
        os.makedirs(os.path.join(output_root, cls), exist_ok=True)

    # 1. Copy cracked images via symlinks
    cracked_in = os.path.join(input_dir, "cracked")
    cracked_files = sorted(os.listdir(cracked_in))
    for fn in tqdm(cracked_files, desc="Copying cracked images"):
        src = os.path.join(cracked_in, fn)
        dst = os.path.join(output_root, "cracked", fn)
        try:
            os.symlink(os.path.abspath(src), dst)
        except FileExistsError:
            pass

    # 2. Process non-cracked images
    nonc_in = os.path.join(input_dir, "non_cracked")
    nonc_files = sorted(os.listdir(nonc_in))
    random.shuffle(nonc_files)
    split_pt = int(len(nonc_files) * frac_corrosion)
    corrosion_files = nonc_files[:split_pt]
    leak_files = nonc_files[split_pt:]

    # Simulate corrosion
    for fn in tqdm(corrosion_files, desc="Simulating corrosion"):
        img = cv2.imread(os.path.join(nonc_in, fn))
        out = add_corrosion(img, texture_path)
        cv2.imwrite(os.path.join(output_root, "corrosion", fn), out)

    # Simulate leak
    for fn in tqdm(leak_files, desc="Simulating leak"):
        img = cv2.imread(os.path.join(nonc_in, fn))
        out = add_leak(img)
        cv2.imwrite(os.path.join(output_root, "leak", fn), out)

    # Summary of generated dataset
    print("\nSynthesis complete:")
    for cls in ["cracked", "corrosion", "leak"]:
        count = len(os.listdir(os.path.join(output_root, cls)))
        print(f"  {cls}: {count} images")


if __name__ == "__main__":
    # Hard-coded paths for convenience
    SDNET_ROOT = "/home/mayur/OP_FractureScope/data/sdnet"
    SYNTH_ROOT = "/home/mayur/OP_FractureScope/data/synth"
    TEXTURE   = "/home/mayur/OP_FractureScope/data/textures/rust_texture.png"
    # Create 50% corrosion, 50% leak from non-cracked
    simulate(SDNET_ROOT, SYNTH_ROOT, TEXTURE, frac_corrosion=0.5)
