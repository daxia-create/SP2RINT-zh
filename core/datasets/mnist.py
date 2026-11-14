# 中文学习注释：保持原始实现不变，仅添加阅读辅助说明。
# 模块路径：core/datasets/mnist.py；关注数据来源、预处理、样本/标签格式、张量形状和数据划分。

from typing import List

import torch
from pyutils.general import logger
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

__all__ = ["MNISTDataset"]


resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


# [中文注释] 定义 MNISTDataset 数据类/对象，封装数据集状态和读取接口。
class MNISTDataset(torch.utils.data.Dataset):
    # [中文注释] 初始化数据集；检查路径、变换、划分方式和默认参数。
    def __init__(
        self,
        root: str,
        split: str,
        train_valid_split_ratio: List[float] = (0.9, 0.1),
        center_crop: bool = True,
        resize: bool = True,
        resize_mode: str = "bicubic",
        binarize: bool = False,
        binarize_threshold: float = 0.1307,
        digits_of_interest: List[float] = list(range(10)),
        n_test_samples: int = 10000,
        n_valid_samples: int = 5000,
    ):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples

        self.load()
        self.n_instance = len(self.data)

    # [中文注释] 读取数据并转换为训练所需格式；核对样本和标签形状。
    def load(self):
        tran = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        if not self.center_crop == 28:
            tran.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 28:
            tran.append(transforms.Resize(self.resize, interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = datasets.MNIST(
                self.root, train=True, download=True, transform=transform
            )
            idx, _ = torch.stack(
                [train_valid.targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            train_valid.targets = train_valid.targets[idx]
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                if self.n_valid_samples is None:
                    # use all samples in valid set
                    self.data = valid_subset
                else:
                    # use a subset of valid set, useful to speedup evo search
                    valid_subset.indices = valid_subset.indices[: self.n_valid_samples]
                    self.data = valid_subset
                    logger.warning(
                        f"Only use the front "
                        f"{self.n_valid_samples} images as "
                        f"VALID set."
                    )

        else:
            test = datasets.MNIST(self.root, train=False, transform=transform)
            idx, _ = torch.stack(
                [test.targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            test.targets = test.targets[idx]
            test.data = test.data[idx]
            if self.n_test_samples is None:
                # use all samples as test set
                self.data = test
            else:
                # use a subset as test set
                test.targets = test.targets[: self.n_test_samples]
                test.data = test.data[: self.n_test_samples]
                self.data = test
                logger.warning(
                    f"Only use the front {self.n_test_samples} " f"images as TEST set."
                )

    # [中文注释] 返回单个样本及标签；确认数据类型、维度和变换顺序。
    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1.0 * (img > self.binarize_threshold) + -1.0 * (
                img <= self.binarize_threshold
            )
        digit = self.digits_of_interest.index(self.data[index][1])
        return img, torch.tensor(digit).long()
        # instance = {'image': img, 'digit': digit}
        # return instance

    # [中文注释] 返回数据集样本数量。
    def __len__(self) -> int:
        return self.n_instance


# [中文注释] 实现 test；结合调用处确认数据流及预处理含义。
def test():
    mnist = MNISTDataset(
        root="../../data",
        split="train",
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=28,
        resize=28,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=(3, 6),
        n_test_samples=100,
        n_valid_samples=1000,
    )
    data, label = mnist.__getitem__(20)
    print(data.size(), label.size())
    print("finish")


if __name__ == "__main__":
    test()
