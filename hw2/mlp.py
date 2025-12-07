import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]

        # TODO:
        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.
        # ====== YOUR CODE: ======
        super().__init__()
        prev_d = in_dim
        layers = []
        non_linears = []
        for dim, nonlin in zip(dims, nonlins):
            linear = nn.Linear(prev_d, dim)
            layers.append(linear)
            if isinstance(nonlin, str):
                if nonlin not in ACTIVATIONS:
                    raise ValueError(f"Unknown activation name: {nonlin}")
                act_cls = ACTIVATIONS[nonlin]
                act_kwargs = ACTIVATION_DEFAULT_KWARGS[nonlin]
                act = act_cls(**act_kwargs)
            elif isinstance(nonlin, nn.Module):
                act = nonlin
            else:
                raise TypeError(
                    f"nonlin must be str or nn.Module, got {type(nonlin)}"
                )

            non_linears.append(act)
            prev_d = dim
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(non_linears)
        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        # TODO: Implement the model's forward pass. Make sure the input and output
        #  shapes are as expected.
        # ====== YOUR CODE: ======

        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))

        return x

        # ========================
