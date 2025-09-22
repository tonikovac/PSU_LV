import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

train_dir = "./gtsrb/Train"

classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"Broj klasa: {len(classes)}")
print("Klase u Training direktoriju:")
print(classes[:10], "...")

print("\nBroj slika po klasi:")
for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    num_images = len([f for f in os.listdir(cls_path) if f.lower().endswith('.png')])
    print(f"Klasa {cls}: {num_images} slika")

example_class = classes[0]
example_dir = os.path.join(train_dir, example_class)

all_images = [f for f in os.listdir(example_dir) if f.lower().endswith('.png')]

if len(all_images) == 0:
    print(f"Nema slika u klasi {example_class}. Provjeri strukturu direktorija!")
else:
    sample_images = random.sample(all_images, min(5, len(all_images)))

    plt.figure(figsize=(10, 2))
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(example_dir, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Klasa {example_class}")
    plt.tight_layout()
    plt.show()
