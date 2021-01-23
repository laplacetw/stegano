#!usr/bin/env python3
import cv2
import math
import hashlib
import numpy as np
from functools import reduce

def global_feature(auth_data):
    md5 = hashlib.md5()
    md5.update(auth_data.encode('utf-8'))
    md5 = md5.hexdigest()
    md5 = f'{int(md5, 16):0128b}'
    md5 = [int(md5[i:i+8], 2) for i in range(0, 128, 8)]
    return reduce(lambda x, y: x ^ y, md5)

def local_feature(i, j, block):
    feature = []
    for row in block:
        for col in row:
            rgb = f'{col[0]:08b}'[:5] + f'{col[1]:08b}'[:5] + f'{col[2]:08b}'[:5]
            # parity check bit
            rgb += '1' if rgb.count('1') % 2 == 1 else '0'
            feature.append(int(rgb, 2))

    feature = reduce(lambda x, y: x ^ y, feature)
    return feature ^ i ^ j  

def block_mapping(img_rgb, img_len):
    key = np.array([
        [5, 0],
        [7, 7]
    ])

    bias = np.array([
        [1],
        [1]
    ])

    i, j = 1, 1
    map_seq = []
    for block in img_rgb:
        if j > img_len:
            j = 1
            i += 1
        
        idx = np.array([
            [i],
            [j]
        ])
        seq = (np.dot(key, idx) % img_len + bias).reshape(2)
        u, v = seq[0] - 1, seq[1] - 1
        map_seq.append([u, v])
        j += 1
    
    return map_seq

def gen_recovery_data(block, idx, seq):
    r, g, b = 0, 0, 0
    params = np.array([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.5],
        [0.5, -0.418, -0.082]
    ])

    for row in block:
        for col in row:
            r += col[0]
            g += col[1]
            b += col[2]
    
    RGB = np.array([
        [r / 4],
        [g / 4],
        [b / 4]
    ])

    bias = np.array([
        [0],
        [128],
        [128]
    ])

    Y, Cb, Cr = np.uint8(np.dot(params, RGB) + bias).reshape(3)
    return f'{Ｙ:08b}' + f'{Cb:08b}'[:6] + f'{Cr:08b}'[:6]

def embed(block, w):
    w = [w[i:i+9] for i in range(0, 36, 9)]
    w = np.array(w).reshape(2, 2)

    for idx_r, row in enumerate(block):
        for idx_c, col in enumerate(row):
            data = w[idx_r][idx_c]
            r, g, b = col[0], col[1], col[2]
            r = int(f'{r:08b}'[:5] + data[0:3], 2)
            g = int(f'{g:08b}'[:5] + data[3:6], 2)
            b = int(f'{b:08b}'[:5] + data[6:9], 2)
            block[idx_r][idx_c] = [r, g, b]

# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def psnr(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    
    return 10 * math.log10(255.0 ** 2 / mse)

# divided into non-overlapping 2x2 blocks
def img_split(img_rgb):
    img_len = 128
    img_rgb = np.split(img_rgb, img_len)
    img_rgb = np.array([np.split(x, img_len, 1) for x in img_rgb])
    img_rgb = img_rgb.reshape(-1, 2, 2, 3)
    return img_rgb, img_len

# rebuild 256x256 image
def img_reshape(img_rgb, channel=3):
    new = np.zeros((256, 256, channel)).astype(int)
    img_len = new.shape[0]
    new_idx = list(range(0, img_len, 2))

    count = 0
    for i in new_idx:
        for j in new_idx:
            new[i][j] = img_rgb[count][0][0]
            new[i][j + 1] = img_rgb[count][0][1]
            new[i + 1][j] = img_rgb[count][1][0]
            new[i + 1][j + 1] = img_rgb[count][1][1]
            count += 1
    
    return new

def img_read(path):
    img_bgr = cv2.imread(path)
    b, g, r = cv2.split(img_bgr)
    img_rgb = cv2.merge([r,g,b])  # bgr -> rgb
    return img_split(img_rgb)

def img_write(path, img_rgb):
    img_rgb = img_reshape(img_rgb)
    r, g, b = cv2.split(img_rgb)
    img_bgr = cv2.merge([b, g, r])  # rgb -> bgr
    cv2.imwrite(path, img_bgr)

def gen_anti_tampered_img(g_feature, path, new_path):
    img_rgb, img_len = img_read(path)
    map_seq = block_mapping(img_rgb, img_len)

    # generate & embed watermark (36-bit)
    for idx, block in enumerate(img_rgb):
        u, v = map_seq[idx]
        seq = u * img_len + v
        l_feature = local_feature(u, v, img_rgb[seq])
        auth_data = f'{l_feature ^ g_feature:016b}'
        rec_data = gen_recovery_data(block, idx, seq)
        watermark = auth_data + rec_data
        embed(img_rgb[seq], watermark)

    img_write(new_path, img_rgb)

def extract_g_feature(img_rgb, img_len):
    i, j = 0, 0
    g_feature_stat = {}
    g_feature_list = []
    for idx, block in enumerate(img_rgb):
        if j == img_len:
                j = 0
                i += 1
        
        tmp = ""
        for row in block:
            for col in row:
                for val in col:
                    tmp += f'{val:08b}'[-3:]

        auth_data = int(tmp[:16], 2)
        l_feature = local_feature(i, j, block)
        g_feature = auth_data ^ l_feature
        g_feature_list.append(g_feature)

        if  g_feature_stat.get(g_feature):
            g_feature_stat[g_feature] += 1
        else:
            g_feature_stat[g_feature] = 1

        j += 1

    # find g_feature which has maximum occurrence frequency
    g_feature_stat = sorted(g_feature_stat.items(), key=lambda x:x[1], reverse=True)
    return next(iter(g_feature_stat))[0], g_feature_list  # (first key of dict, g_feature_list list)

def temper_detection(path, new_path):
    img_rgb, img_len = img_read(path)
    g_feature_std, g_feature_list = extract_g_feature(img_rgb, img_len)

    temper_detect = np.full((256, 256), 0)
    temper_detect = np.split(temper_detect, 128)
    temper_detect = np.array([np.split(x, 128, 1) for x in temper_detect])
    temper_detect = temper_detect.reshape(-1, 2, 2)

    for idx_r, row in enumerate(temper_detect):
        if g_feature_list[idx_r] != g_feature_std:
            for col in row:
                for idx_c, _ in enumerate(col):
                    col[idx_c] = 255

    temper_detect = img_reshape(temper_detect, 1)
    cv2.imwrite(new_path, temper_detect)

def gen_recovery_img(path, new_path):
    img_rgb, img_len = img_read(path)
    map_seq = block_mapping(img_rgb, img_len)

    img_size = len(img_rgb)
    g_feature_std, g_feature_list = extract_g_feature(img_rgb, img_len)
    for idx_r, row in enumerate(img_rgb):
        if g_feature_list[idx_r] != g_feature_std:
            u, v = map_seq[idx_r]
            seq = u * img_len + v

            # non-tampered recovery data
            if g_feature_list[seq] == g_feature_std:
                w_data = ""
                for row_s in img_rgb[seq]:
                    for col_s in row_s:
                        r, g, b = col_s[0], col_s[1], col_s[2]
                        r = f'{r:08b}'[-3:]
                        g = f'{g:08b}'[-3:]
                        b = f'{b:08b}'[-3:]
                        w_data += r + g + b

                r_data = w_data[-20:]
                Y_Cb_Cr = np.array([
                    [int(r_data[:8], 2)],
                    [(int(r_data[8:14] + '00', 2) - 128)],
                    [(int(r_data[14:] + '00', 2) - 128)]
                ])

                params = np.array([
                    [1, 0, 1.402],
                    [1, -0.344, -0.714],
                    [1, 1.772, 0]
                ])

                RGB = (np.dot(params, Y_Cb_Cr)).reshape(3)
                np.putmask(RGB, RGB > 255, 255)
                np.putmask(RGB, RGB < 0, 0)
                RGB = np.uint8(RGB)

                for idx_c, col in enumerate(row):
                    for idx_p, _ in enumerate(col):
                        col[idx_p] = RGB
            else:
                blocks = list(range(8))
                blocks[0] = idx_r - 1        # left
                blocks[1] = idx_r + 1        # right
                blocks[2] = idx_r - img_len  # up
                blocks[3] = blocks[2] - 1    # up_left
                blocks[4] = blocks[2] + 1    # up_right
                blocks[5] = idx_r + img_len  # down
                blocks[6] = blocks[5] - 1    # down_left
                blocks[7] = blocks[5] + 1    # down_right

                count = 0
                block_avg = np.zeros((2, 2, 3), int)
                
                for idx in blocks:
                    if idx >= 0 and idx < img_size:
                        count += 1
                        block_avg += img_rgb[idx]

                block_avg = block_avg / count
                img_rgb[idx_r] = block_avg

    img_write(new_path, img_rgb)

g_feature = global_feature("Jan. 23, 2021; 09:41 AM, laplacetw")
gen_anti_tampered_img(g_feature, "img1.png", "img1_embed.png")
#temper_detection("img1_tamper.png", "img1_detect.png")
#gen_recovery_img("img1_tamper.png", "img1_recovery.png")
#print("Embeded PSNR=", '%.2f' % psnr("img1.png", "img1_embed.png"), "dB")
#print("Recovery PSNR=", '%.2f' % psnr("img1.png", "img1_recovery.png"), "dB")