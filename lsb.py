#!usr/bin/env python3
# coding:utf-8
import cv2
import math
import time
import numpy as np

def quantize(img, q):
    img_q = img.copy()
    for idx_r, row in enumerate(img_q):
        for idx_c, col in enumerate(row):
            img_q[idx_r][idx_c] = int(col / q) * q
    
    return img_q

def str2bin(msg):
    msg = '$' + msg + '$'  # recognize char
    return "".join(f'{ord(c):08b}' for c in msg)

# https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    
    return 10 * math.log10(255.0 ** 2 / mse)

# q -> quantization value
def img_encrypt(q, codes, origin, encrypt):
    row_size, col_size = 256, 256
    img = cv2.imread(origin, 0)
    if img.shape != (row_size, col_size):
        img = cv2.resize(img, (row_size, col_size))
        cv2.imwrite(origin, img)
    
    encrypted = cv2.imread(origin, 0)
    encrypted = quantize(encrypted, q)

    # embed secret info with LSB(Least Significant Bit)
    idx = 0
    for idx_r, row in enumerate(encrypted):
        for idx_c, col in enumerate(row):
            encrypted[idx_r][idx_c] += int(codes[idx])
            idx += 1

            if idx == len(codes):
                cv2.imwrite(encrypt, encrypted)
                return psnr(img, encrypted)

def img_decrypt(q, encrypt):
    bits = 8
    row_size, col_size = 256, 256
    encrypted = cv2.imread(encrypt, 0)
    encrypted_q = quantize(encrypted, q)

    msg = ""
    tmp = ""
    for row in range(row_size):
        for col in range(col_size):
            code = encrypted[row, col] - encrypted_q[row, col]
            tmp += str(code)

            if len(tmp) == bits:
                num = int(tmp, 2)
                c = chr(num)
                tmp = ""
                
                if c == '$' or len(msg) > 0:
                    msg += c

                if len(msg) > 1 and msg[len(msg) - 1] == '$':
                    return msg.replace('$', '').replace('     ', ' ')

secret = "After billions of dollars and a decade of work, NASA's plans to send astronauts \
    back to the moon had a new setback on Saturday. A planned eight-minute test firing of \
    the four engines of a new mega rocket needed for the moon missions came to an abrupt \
    end after only about a minute.As engineers disentangle what went wrong, the first \
    launch of the rocket is likely to slip further into the future, and NASA astronauts \
    may have to wait longer before setting foot on the moon again."

quantization = input("Input 'L': ")
origin = 'origin.png'
encrypt = 'encrypt_L' + quantization + '.png'

q = int(quantization)
codes = str2bin(secret)
p = img_encrypt(q, codes, origin, encrypt)
print(encrypt, 'PSNR: %.2f dB' % p)

time.sleep(1)
msg = img_decrypt(q, encrypt)
print("\nsecret message in ", encrypt, ":\n" + msg)