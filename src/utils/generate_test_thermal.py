import os
import cv2

def simulate_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)  # You can try COLORMAP_INFERNO too
    return thermal

def process_class(class_dir):
    visual_dir = os.path.join(class_dir, 'visual')
    thermal_dir = os.path.join(class_dir, 'thermal')

    os.makedirs(thermal_dir, exist_ok=True)

    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)

        # Skip directories or already processed ones
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')) or not os.path.isfile(img_path):
            continue

        # Move original image to 'visual'
        new_visual_path = os.path.join(visual_dir, filename)
        os.makedirs(visual_dir, exist_ok=True)
        os.rename(img_path, new_visual_path)

        # Create thermal version
        img = cv2.imread(new_visual_path)
        thermal = simulate_thermal(img)

        thermal_path = os.path.join(thermal_dir, filename)
        cv2.imwrite(thermal_path, thermal)

    print(f"[DONE] Processed: {class_dir}")

if __name__ == "__main__":
    base_dir = "/home/mayur/OP_FractureScope/data/synth/test_dataset"
    classes = ['cracked', 'corrosion', 'leak']

    for cls in classes:
        process_class(os.path.join(base_dir, cls))
