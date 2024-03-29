import struct
import numpy as np
import random


def one_hot(index):
    one_hot = np.zeros((10, 1))
    one_hot[index] = 1.0
    return one_hot


def read(dataset="training", path="mnist/"):
    if dataset is "training":
        fname_img = path + 'train-images.idx3-ubyte'
        fname_lbl = path + 'train-labels.idx1-ubyte'
    elif dataset is "testing":
        fname_img = path + 't10k-images.idx3-ubyte'
        fname_lbl = path + 't10k-labels.idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, "rb") as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def show(image):
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


training = list(read(dataset='training'))
test = list(read(dataset='testing'))


def next_batch(batch_size, current, dataset="training"):
    if current == 0:
        random.shuffle(training)
    if dataset == 'training':
        data = training[current:current+batch_size]
    else:
        data = test[current:current+batch_size]

    images = np.stack(np.array(v[1]) for v in data)/256
    labels = np.stack(one_hot(v[0]) for v in data)
    current = current + batch_size
    return images, labels, current



if __name__ == '__main__':
    images, _, _ = next_batch(1, 0)

