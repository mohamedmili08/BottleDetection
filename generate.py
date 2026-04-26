import os
import cv2
import albumentations as A
from glob import glob

base_dir = "dataset"

folder_map = {
    "Real-OK": "augmented-ok",
    "Real-NoCap": "augmented-nocap",
    "Real-Damaged": "augmented-damaged"
}

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(-0.03, 0.03),
        rotate=(-5, 5),
        shear=(-2, 2),
        p=0.7
    )
])

copies_per_image = 9

for real_folder, aug_folder in folder_map.items():
    input_folder = os.path.join(base_dir, real_folder)
    output_folder = os.path.join(base_dir, aug_folder)

    os.makedirs(output_folder, exist_ok=True)

    image_paths = []
    image_paths += glob(os.path.join(input_folder, "*.jpg"))
    image_paths += glob(os.path.join(input_folder, "*.jpeg"))
    image_paths += glob(os.path.join(input_folder, "*.png"))

    print(f"\nProcessing folder: {real_folder}")
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        base_name, _ = os.path.splitext(image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read: {image_path}")
            continue

        # حفظ نسخة من الأصل داخل مجلد augmented
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_orig.jpg"), image)

        # إنشاء نسخ جديدة
        for i in range(copies_per_image):
            augmented = transform(image=image)
            aug_image = augmented["image"]

            save_path = os.path.join(output_folder, f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(save_path, aug_image)

    print(f"Saved augmented images to: {output_folder}")

print("\nDone! All folders processed successfully.")