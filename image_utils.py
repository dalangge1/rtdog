from multiprocessing import Queue, Process
from itertools import islice
import os
import numpy as np
from PIL import Image



def get_color_distrib(path, nproc):
    '''Returns mean and std dev for R, G and B for all the images in the path'''

    def worker(filenames, out_q):
        mean_sum = np.zeros(3)
        var_sum = np.zeros(3)
        i = 0
        for fn in filenames:
            i += 1
            img = np.array(Image.open(fn))
            mean_sum += np.mean(img, (0, 1))
            var_sum += np.mean(np.square(img, dtype=np.uint16), (0, 1))

        mean = mean_sum / i
        var = var_sum / (i - 1) - np.square(mean)
        # std_dev = np.sqrt(std_dev)

        out_q.put((mean, var))

    out_q = Queue()
    num_files = len([f for f in os.listdir(path)])
    chunksize = int(num_files / float(nproc))
    procs = []

    for i in range(nproc):
        iter = islice(map(lambda f: os.path.join(path, f), os.listdir(path)), i*chunksize, (i+1)*chunksize)
        p = Process(target=worker,
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

    for p in procs:
        p.join()

    return full_mean, full_dev

import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = np.array(Image.open('dog.jpg'))
    for i in range(2):
        #img[:,:,i] -= 110
        img = np.divide(img, 65)

    print(img)
    plt.imshow(img)
    plt.show()

