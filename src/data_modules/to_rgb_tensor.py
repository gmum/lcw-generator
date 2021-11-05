import torch


class ToRgbTensor:
    def __call__(self, pic: torch.Tensor):
        pic = pic.repeat(3, 1, 1)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToLongTensor:
    def __call__(self, pic: torch.Tensor):
        pic = (pic * 255).byte()
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '()'
