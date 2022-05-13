import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)  # 返回随机数/数组(整数)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)  # 截取函数
            y2 = np.clip(y + self.length // 2, 0, h)  # 用于截取数组中小于或者大于某值的部分，
            x1 = np.clip(x - self.length // 2, 0, w)  # 并使得被截取的部分等于固定的值
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)  # 数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        mask = mask.expand_as(img)  # 把一个tensor变成和函数括号内一样形状的tensor
        img = img * mask

        return img


im1 = Image.imread("C:/Users/guiziqiu/Desktop/pic3.png")
cut = Cutout(1, 5)
im1 = torch.tensor(im1)
img1 = cut.__call__(im1)
img1=img1.numpy()
plt.imshow(img1)
plt.show()