import os
import random
import shutil
import pygame
import cv2
import numpy as np

import config as cg
import dataset_service as s


random.seed(cg.dataset_seed)
back_color = (0, 0, 0)
front_color = (255, 255, 255)

print()
with open(cg.fonts_file, 'r') as file:
    fonts = [line.strip() for line in file if line.strip()]
all_available_fonts = pygame.font.get_fonts()
unavailable_fonts = [f for f in fonts if (f not in all_available_fonts)]
fonts = [f for f in fonts if (f not in unavailable_fonts)]
fonts_len = len(fonts)
print(f"Fonts found: {fonts_len}/{fonts_len + len(unavailable_fonts)}")
if len(unavailable_fonts) != 0:
    print("Unavailable fonts:")
    for font in unavailable_fonts:
        print(font)
print()

if not os.path.exists(cg.dataset_folder):
    os.makedirs(cg.dataset_folder)
else:
    shutil.rmtree(cg.dataset_folder)

pygame.init()

if sorted("".join(cg.chars_types)) != sorted(cg.char_labels):
    raise Exception("Line 37.")
counters = [0] * len(cg.char_labels)

# t0-a, t1-x, t2-y, t3-F, t4-X, t5-j
t0, t1, t2, t3, t4, t5 = cg.chars_types
modes = [[t0, t5], [t0, t2], [t0, t3], [t0],
         [t1, t5], [t1, t3],
         [t2, t3], [t2],
         [t3, t2], [t3],
         [t4, t2], [t4],
         [t5]]
for font_i, font_v in enumerate(fonts, start=1):
    print(f"{font_i}/{fonts_len} - {font_v}")
    for mode_i, mode_v in enumerate(modes):
        for size_factor in cg.size_factors:
            size = int(size_factor * cg.image_size)
            chars_to_draw = "".join(mode_v)
            screen = pygame.Surface((len(chars_to_draw) * size, size * 2))

            screen.fill(back_color)
            selected_font = pygame.font.SysFont(font_v, size)

            for char_i, char_v in enumerate(chars_to_draw):
                font_size = selected_font.size(char_v)
                text = selected_font.render(char_v, True, front_color, back_color)
                screen.blit(text, (char_i * size, 0.25 * size))

            image = pygame.surfarray.array_red(screen).transpose()
            image = s.crop_image(image, axis=0)
            for char_i, char_v in enumerate(mode_v[0]):
                char = image[:, char_i*size:(char_i+1)*size]
                char = s.crop_image(char, axis=1)
                char = s.make_square(char)
                char = s.resize(char, cg.image_size)
                _, char = cv2.threshold(char, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                index = cg.char_labels.index(char_v)
                img_name = font_v + "_" + str(mode_i) + "_" + str(size_factor) + ".png"
                img_path = os.path.join(cg.dataset_folder, str(index))
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                counters[index] += 1
                cv2.imwrite(os.path.join(img_path, img_name), char)

print()
for char_i, char_v in enumerate(cg.char_labels):
    unique_chars = []
    unique_chars_names = []
    folder_path = os.path.join(cg.dataset_folder, str(char_i))
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.tolist()
        if image not in unique_chars:
            unique_chars.append(image)
            unique_chars_names.append(file_name)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    for image, file_name in zip(unique_chars, unique_chars_names):
        image_path = os.path.join(folder_path, file_name)
        cv2.imwrite(image_path, np.array(image))
    print("{:>3}".format(char_i), '-', char_v, '-', f"{counters[char_i] - len(unique_chars)} removed - {len(unique_chars)}")
    counters[char_i] = len(unique_chars)

print()
min_counter = min(counters)
for char_i, char_v in enumerate(cg.char_labels):
    to_remove = counters[char_i] - min_counter
    folder_path = os.path.join(cg.dataset_folder, str(char_i))
    files = os.listdir(folder_path)
    random_files = random.sample(files, to_remove)
    for file_name in random_files:
        file_name = os.path.join(folder_path, file_name)
        os.remove(file_name)
    print("{:>3}".format(char_i), '-', char_v, '-', f"{to_remove} removed - {counters[char_i] - to_remove}")
    counters[char_i] = counters[char_i] - to_remove

print("\nCompleted.")
