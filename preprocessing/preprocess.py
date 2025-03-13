import cv2
import os
from PIL import Image, ImageOps
import random
import numpy as np
from tqdm import tqdm

### SET THESE VARIABLES ***
crosstalk_threshold = 8
frames_dir = "/media/nick/C8EB-647B/Data/processed/2_chs/frames/Exp_1/"
output_dir = "/media/nick/C8EB-647B/Data/processed/2_chs/augmented/Exp_1/"

CS_IDX = 0
MA_IDX = 1

##########################

to_iter = sorted(os.listdir(frames_dir))

cs = to_iter[CS_IDX]
ma = to_iter[MA_IDX]

def remove_crosstalk(img):
    return Image.fromarray(cv2.threshold(img, crosstalk_threshold, 255, cv2.THRESH_TOZERO)[1]) 

def toroidal_translate(image, tx, ty):
    """
    Apply toroidal translation to an image using numpy for array manipulation.
    Converts the PIL Image to a numpy array, applies the translation, and then converts it back.
    """
    image_array = np.array(image)
    translated_image_array = np.roll(image_array, shift=tx, axis=1)  # Horizontal wrap
    translated_image_array = np.roll(translated_image_array, shift=ty, axis=0)  # Vertical wrap
    return Image.fromarray(translated_image_array)

def tile_and_shuffle(image, tiles_x, tiles_y, shuffle_order):
    """
    Shuffles the image tiles according to the given shuffle order.
    """
    width, height = image.size
    tile_width = width // tiles_x
    tile_height = height // tiles_y
    tiles = []

    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            box = (x, y, x + tile_width, y + tile_height)
            tiles.append(image.crop(box))

    shuffled_image = Image.new('L', (width, height))
    for idx, order in enumerate(shuffle_order):
        x = (order % tiles_x) * tile_width
        y = (order // tiles_x) * tile_height
        shuffled_image.paste(tiles[idx], (x, y))
    return shuffled_image


for jdx in range(len(cs)):
    cs_file = cs[jdx]
    ma_file = ma[jdx]
    print(f"doing files {cs_file} and {ma_file}")


    cs_out = os.path.join(output_dir, cs_file)
    ma_out = os.path.join(output_dir, ma_file)
    # overlay_out = os.path.join(output_dir, f"overlay{jdx+3}")

    print(cs_out)
    print(ma_out)
    # print(overlay_out)

    os.makedirs(cs_out, exist_ok=True)
    os.makedirs(ma_out, exist_ok=True)
    # os.makedirs(overlay_out, exist_ok=True)

    cs_files = sorted(os.listdir(os.path.join(frames_dir, cs_file)))
    ma_files = sorted(os.listdir(os.path.join(frames_dir, ma_file)))


    # Shuffle the indices
    indices = list(range(len(cs_files)))
    random.shuffle(indices)

    # Select the shuffled indices
    selected_indices = indices    
    loop = tqdm(selected_indices)
    for idx in loop:
        cs_file_sample = cs_files[idx]
        ma_file_sample = ma_files[idx]
        # print(os.path.join(frames_dir, cs_file_sample, cs_files[idx]))
        img_cs = cv2.imread(os.path.join(frames_dir, cs[jdx], cs_files[idx]), cv2.IMREAD_GRAYSCALE)
        img_ma = cv2.imread(os.path.join(frames_dir, ma[jdx], ma_files[idx]), cv2.IMREAD_GRAYSCALE)

        img_cs = cv2.resize(img_cs, (300, 240))
        img_ma = cv2.resize(img_ma, (300, 240))


        img_cs = remove_crosstalk(img_cs)
        img_ma_dn = Image.fromarray(img_ma)

        tx = random.randint(-150, 150)
        ty = random.randint(-120, 120)

        img_ma_dn = toroidal_translate(img_ma_dn, tx, ty)
        img_cs = toroidal_translate(img_cs, tx, ty)

        tiles_side = 5
        total_tiles = tiles_side ** 2
        shuffle_order = list(range(total_tiles))
        random.shuffle(shuffle_order)

        ma_dn_shuffled = tile_and_shuffle(img_ma_dn, tiles_side, tiles_side, shuffle_order)
        cs_shuffled = tile_and_shuffle(img_cs, tiles_side, tiles_side, shuffle_order)


        cs_shuffled.save(os.path.join(cs_out, f"frame_{idx:05d}"), "PNG")
        ma_dn_shuffled.save(os.path.join(ma_out, f"frame_{idx:05d}"), "PNG")



