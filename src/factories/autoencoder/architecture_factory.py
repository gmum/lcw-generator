from torch.nn import Module
from architecture.encoder.mnist_encoder import Encoder as MnistEncoder
from architecture.decoder.mnist_decoder import Decoder as MnistDecoder
from architecture.encoder.celeba_encoder import Encoder as CelebaEncoder
from architecture.decoder.celeba_decoder import Decoder as CelebaDecoder
from architecture.encoder.svhn_encoder import Encoder as SvhnEncoder
from architecture.decoder.svhn_decoder import Decoder as SvhnDecoder
from architecture.encoder.stacked_mnist_encoder import Encoder as StackedMnistEncoder
from architecture.decoder.stacked_mnist_decoder import Decoder as StackedMnistDecoder


def get_architecture(identifier: str, z_dim: int) -> tuple[Module, Module]:
    if identifier == 'mnist' or identifier == 'fmnist' or identifier == 'kmnist':
        return MnistEncoder(z_dim), MnistDecoder(z_dim)

    if identifier == 'celeba':
        return CelebaEncoder(z_dim), CelebaDecoder(z_dim)

    if identifier == 'svhn':
        return SvhnEncoder(z_dim), SvhnDecoder(z_dim)

    if identifier == 'stacked_mnist':
        return StackedMnistEncoder(z_dim), StackedMnistDecoder(z_dim)

    raise ValueError("Unknown architecture")
