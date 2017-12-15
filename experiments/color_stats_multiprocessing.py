# get the color mean and std dev of the data with multiple processes and compare speed

from multiprocessing import Queue, Process
from itertools import islice
import os
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

def color_worker(filenames, out_q):
    mean_sum = np.zeros(3)
    var_sum = np.zeros(3)
    i = 0
    for fn in filenames:
        i += 1
        img = np.array(Image.open(fn))
        mean_sum += np.mean(img, (0, 1))
        var_sum += np.mean(np.square(img, dtype=np.uint16), (0, 1))

    mean = mean_sum / i
    var = var_sum / (i-1) - np.square(mean)
    # std_dev = np.sqrt(std_dev)

    out_q.put((mean, var))

def run_workers(path, nproc):

    out_q = Queue()
    num_files = len([f for f in os.listdir(path)])
    chunksize = int(num_files / float(nproc))
    procs = []

    start_time = time.time()

    for i in range(nproc):
        iter = islice(map(lambda f: os.path.join(path, f), os.listdir(path)), i*chunksize, (i+1)*chunksize)
        p = Process(target=color_worker,
                    args=(iter, out_q))
        procs.append(p)
        p.start()

    full_mean = np.zeros(3)
    full_dev = np.zeros(3)

    for i in range(nproc):
        m, d = out_q.get()
        full_mean += m
        full_dev += d


    full_mean /= nproc
    full_dev /= nproc

    full_dev = np.sqrt(full_dev)


    print(full_dev)
    print(full_mean)
    for p in procs:
        p.join()

    time_taken = (time.time() - start_time) * 1000
    return time_taken
    print("{} workers : {:0.5f} ms (avg {:0.5f})".format(nproc, time_taken, time_taken / num_files))

if __name__ == '__main__':
    times = []
    for workers in range(1, 8):
        times.append(run_workers('../data/train', workers))

    plt.plot(range(1, 8), times)
    plt.show()