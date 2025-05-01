import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

dataset_path = './'

categories = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

print('이미지 개수 [클래스별]:')
for category in categories:
    category_path = os.path.join(dataset_path, category)
    image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
    print(f"- {category}: {len(image_files)}장")

print("\n" + "="*50 + "\n")

#랜덤으로 샘플 이미지 띄우기 (각 클래스별 3장씩)
print("샘플 이미지 보기:")

plt.figure(figsize=(15, 10))

for idx, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]

    #3개 랜덤 뽑기
    sample_files = random.sample(image_files, 3)

    for j, sample_file in enumerate(sample_files):
        img_path = os.path.join(category_path, sample_file)
        img = Image.open(img_path).convert('L')

        plt.subplot(len(categories)), 3, idx*3 + j + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{category}\n{sample_file}', fontsize=8)
        plt.axis('off')
plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")

#이미지 크기 확인(몇장 뽑아서 보기)
print("샘플 이미지 크기 확인:")

for category in categories:
    category_path = os.path.join(dataset_path, category)
    image_files = [f for f in os.listdir(category_path) in f.endswith('.jpg')]
    sample_file = random.choice(image_files)

    img.path = os.path.join(category_path, sample_file)
    img = Image.open(img_path)
    print(f"- {category}: {sample_file} -> {img.size}")

print("\n" + "n"*50 + "\n")

#픽셀 값 분포(밝기 히스토그램)