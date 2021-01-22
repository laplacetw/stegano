#!usr/env/bin python3
# coding:utf-8
import cv2
import math
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt

IMG_HEIGHT = 256
IMG_WIDTH = 256
MARK_HEIGHT = 128
MARK_WIDTH = 128
MARK_HEIGHT_HALF = int(MARK_HEIGHT / 2)
MARK_WIDTH_HALF = int(MARK_WIDTH / 2)
OFFSET_w1 = int((IMG_HEIGHT - MARK_HEIGHT) / 2) - 2
OFFSET_w2 = int((IMG_HEIGHT - MARK_HEIGHT) / 2) - 1

def split_watermark(watermark):
    mark = watermark.copy()
    mark = mark.reshape(-1, MARK_HEIGHT_HALF, MARK_WIDTH)
    return mark[0], mark[1]

def insert(origin, watermark, offset_h, offset_w):
    img = origin.copy()
    mark = watermark.copy()
    
    # binarization of 0 or 1
    for r_idx, row in enumerate(mark):
        for c_idx, col in enumerate(row):
            mark[r_idx, c_idx] = 1 if col > 0 else 0
    
    for row in range(0, MARK_HEIGHT_HALF):
        for col in range(0, MARK_WIDTH):
            tmp = img[row + offset_h, col + offset_w] + mark[row, col]
            tmp = 255 if tmp > 255 else tmp
            img[row + offset_h, col + offset_w] = tmp
    
    return img

def extract(new, origin, offset_h, offset_w):
    mark = new - origin
    mark = mark[offset_h:offset_h + MARK_HEIGHT_HALF, offset_w:offset_w + MARK_WIDTH]

    for r_idx, row in enumerate(mark):
        for c_idx, col in enumerate(row):
            mark[r_idx, c_idx] = 255 if col > 0 else 0
    
    return mark

def retrieval(origin, embed_w1, embed_w2):
    # wavelet transform
    LL, (LH, HL, HH) = pywt.dwt2(embed_w1, 'haar')
    _LL, (_LH, _HL, _HH) = pywt.dwt2(embed_w2, 'haar')
    # extract part 1 of watermark
    extract_w1 = extract(embed_w2, origin, OFFSET_w1, OFFSET_w1)
    # extract part 2 of watermark
    extract_w2 = extract(_HH, HH, OFFSET_w2, 0)
    extract_w = np.concatenate((extract_w1, extract_w2))

    return extract_w1, extract_w2, extract_w

# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    
    return 10 * math.log10(255.0 ** 2 / mse)

def cut_rectangle(image, h, w):
    img = image.copy()
    offset_h = int((IMG_HEIGHT - h) / 2)
    offset_w = int((IMG_WIDTH - w) / 2)
    
    for row in range(h):
        for col in range(w):
            img[(row + offset_h), (col + offset_w)] = None
    
    return img

def cut_box_shape(image):
    img = image.copy()
    lw = 16  # line width
    lw_half = int(lw / 2)
    offset_h = int(IMG_HEIGHT / 4)
    offset_w = int(IMG_WIDTH / 4)
    h1, w1 = IMG_HEIGHT - offset_h * 2, IMG_WIDTH - offset_w * 2
    h2, w2 = int(h1 / 2), int(w1 / 2)
    h2_half, w2_half = int(h2 / 2), int(w2 / 2)

    layer1 = np.zeros((h1, w1))
    layer2 = np.full((h1 - lw, w1 - lw), 255)
    layer3 = np.zeros((h2, w2))
    layer4 = np.full((h2 - lw, w2 - lw), 255)

    layer1[lw_half:h1 - lw_half, lw_half:w1 - lw_half] = layer2
    layer3[lw_half:h2 - lw_half, lw_half:w2 - lw_half] = layer4
    layer1[h2_half:h1 - h2_half, w2_half:w1 - w2_half] = layer3

    mask = np.full((IMG_HEIGHT, IMG_WIDTH), 255)
    mask[offset_h:IMG_HEIGHT - offset_h, offset_w:IMG_WIDTH - offset_w] = layer1
    
    for r_idx, row in enumerate(mask):
        for c_idx, col in enumerate(row):
            img[r_idx, c_idx] = None if col == 0 else img[r_idx, c_idx]

    return img

# Load image
origin = cv2.imread('origin.png', 0)
watermark = cv2.imread('watermark.png', 0)
_, watermark = cv2.threshold(watermark, 128, 255, cv2.THRESH_OTSU)
w1, w2 = split_watermark(watermark)

img_embed_w1 = insert(origin, w1, OFFSET_w1, OFFSET_w1)
# wavelet transform
LL, (LH, HL, HH) = pywt.dwt2(img_embed_w1, 'haar')
img_embed_HH = insert(HH, w2, OFFSET_w2, 0)
img_embed_w2 = LL, (LH, HL, img_embed_HH)
# inverse wavelet transform
img_embed_w2 = pywt.idwt2(img_embed_w2, 'haar')
print("Image PSNR:", '%.2f' % psnr(img_embed_w2, origin), "dB")

# embed & extract
extract_w1, extract_w2, extract_w = retrieval(origin, img_embed_w1, img_embed_w2)

fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(1, 4, 1)
ax.imshow(img_embed_w1, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(1, 4, 2)
ax.imshow(img_embed_w2, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(2, 4, 3)
ax.imshow(extract_w1, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(2, 4, 7)
ax.imshow(extract_w2, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(1, 4, 4)
ax.imshow(extract_w, interpolation="nearest", cmap=plt.cm.gray)
fig.tight_layout()
plt.savefig('embed_and_extract.png')

# remove 128x128 region from image
cut_128x128 = cut_rectangle(img_embed_w2, 128, 128)
extract_w1, extract_w2, extract_w = retrieval(origin, img_embed_w1, cut_128x128)

fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(1, 3, 1)
ax.imshow(cut_128x128, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(2, 3, 2)
ax.imshow(extract_w1, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(2, 3, 5)
ax.imshow(extract_w2, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(1, 3, 3)
ax.imshow(extract_w, interpolation="nearest", cmap=plt.cm.gray)
fig.tight_layout()
plt.savefig('remove_128x128.png')

# remove box shape from image
cut_bs = cut_box_shape(img_embed_w2)
extract_w1, extract_w2, extract_w = retrieval(origin, img_embed_w1, cut_bs)

fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(1, 3, 1)
ax.imshow(cut_bs, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(2, 3, 2)
ax.imshow(extract_w1, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(2, 3, 5)
ax.imshow(extract_w2, interpolation="nearest", cmap=plt.cm.gray)

ax = fig.add_subplot(1, 3, 3)
ax.imshow(extract_w, interpolation="nearest", cmap=plt.cm.gray)
fig.tight_layout()
plt.savefig('remove_box_shape.png')