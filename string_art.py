from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
from tqdm import tqdm
from skimage import draw
from pyembroidery import *
import cv2

import time
import functools


def embroider(final_line):
    pattern = EmbPattern()

    pattern.add_block([line[0] for line in final_line], "black")

    pattern.move_center_to_origin()

    write_dst(pattern, "out.dst", {"long_stitch_contingency":CONTINGENCY_LONG_STITCH_SEW_TO, "max_stitch":5})

def pre_pross():
    #Open Image
    img = Image.open(base_image)
    img = img.resize((500, 500))
    size = img.size

    #Make B&W
    img = img.convert("L")

    #Circle Crop
    mask = Image.new('L', size, 255)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((0, 0) + size, fill=1)
    img = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    img.paste(0, mask=mask)

    #img.show()
    return img

def circle_points(r, n):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = (r - 1) * np.cos(t) + r 
    y = (r - 1) * np.sin(t) + r 
    return np.c_[y.astype(int), x.astype(int)]

def mask_image(size, output):
    cv_img = cv2.imread(base_image)
    cv_img = cv2.resize(cv_img, (size, size))

    mask = np.zeros((cv_img.shape[0] + 2, cv_img.shape[1] + 2),dtype=np.uint8)

    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    cv2.floodFill(cv_img, mask, (0,0), 255, loDiff=(1,1,1,1), upDiff=(1, 1, 1, 1), flags=floodflags)

    inverted_mask = cv2.bitwise_not(mask)

    output = output.resize(mask.shape)

    cv_output = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    cv_output = cv2.cvtColor(cv_output, cv2.COLOR_BGR2GRAY)


    masked_image = cv2.bitwise_and(cv_output, cv_output, mask=inverted_mask)

    out = masked_image + mask

    cv2.imwrite("masked.jpg", out)

    #embroider(final_line)


@functools.cache      
def is_close(n, center, offset, length):
    lo = (center - offset) % length
    hi = (center + offset) % length
    if lo < hi:
        return lo <= n <= hi
    else:
        return n >= lo or n <= hi

def thread_art(img):
    size = img.size
    points = circle_points(size[0] / 2 - 4, nail_count)
    base_point = 0
    final_line = []

    img = np.asarray(img, dtype=np.uint16).copy().T

    for _ in tqdm(range(thread_count), desc=f"Image: {base_image.split('.')[0]} Power: {power} Nail Count: {nail_count} Thread Count: {thread_count}"):
        lines = [(i, draw.line(*points[base_point], *point)) 
                    for i, point in enumerate(points) 
                    if not is_close(i, base_point, min_dist, len(points))]

        adverages = [(np.sum(values := img[line])/len(values), i) for i, line in lines]

        line = min(adverages, key = lambda i : i[0])[1]

        final_line.append((tuple(points[base_point]), tuple(points[line])))

        y, x = draw.line(*points[base_point], *points[line])

        img[y, x] += power
        img[y, x] = img[y, x].clip(max=255)

        base_point = line

    final_line = [[[i * sizing for i in x] for x in line] for line in final_line]

    import pickle
    f = open("sample.pkl", "wb")
    pickle.dump(final_line, f)
    f.close()

    size = size[0]*sizing
    output = Image.new('L', (final_size, final_size), 255)
    out_draw = ImageDraw.Draw(output)

    for line in final_line:
        out_draw.line((*line[0], *line[1]), 0)

    #output.show()

    #output.save(f"out/{base_image.split('.')[0]}_{power}_{nail_count}_{thread_count}.jpg")

    mask_image()

#Variables
base_image = "new.jpg"


thread_count = 600
nail_count = 700
power = 120
min_dist = 150

sizing = 3#2.948#1.474
final_size = 1500

img = pre_pross()

thread_art(img)
