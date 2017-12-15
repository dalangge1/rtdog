# compare speed for opening a simple image

# conclusion : PIL slightly faster
# GFile average time : 0.11386 ms
# PIL average time : 0.10407 ms

import tensorflow as tf
from PIL import Image
import time

RUNS = 5000
filename = '../dog.jpg'


start_time = time.time()
for _ in range(RUNS):
    img = tf.gfile.GFile(filename, 'rb').read()

gfile_time = (time.time() - start_time) / RUNS * 1000

start_time = time.time()
for _ in range(RUNS):
    img = Image.open(filename)

pil_time = (time.time() - start_time) / RUNS * 1000


print('GFile average time : {:0.5} ms\nPIL average time : {:0.5} ms'.format(gfile_time, pil_time))
