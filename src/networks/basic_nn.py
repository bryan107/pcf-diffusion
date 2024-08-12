import torch
from corai_error import Error_type_setter
from corai_util.tools.src import function_iterable
from torch import nn


class BasicNN(nn.Module):
    """
    NN Module where the architecture is set up as:

    Input
        -> Hidden Layer[0] -> activation function [0] - (no dropout on first layer)
        -> Hidden Layer[1] -> Dropout -> activation function [1]
        -> ...
        -> Hidden Layer[-1] -> Output
    To apply a last activation function, do it outside.
    By passing list_hidden_sizes = [0], the model is a simple linear model and the activation function remains unused.
    """

    module_linearity = nn.Linear

    def __init__(
            self,
            input_size,
            list_hidden_sizes,
            output_size,
            list_biases,
            activation_functions,
            dropout,
    ):
        super().__init__()

        self.input_size = input_size
        self.list_hidden_sizes = list_hidden_sizes
        self.is_model_without_hidden_layers = not self.list_hidden_sizes[0]

        self.output_size = output_size
        # should always be defined after list_hidden_sizes.
        self.biases = list_biases
        self.activation_functions = activation_functions
        self.dropout = dropout

        self.set_layers()

    def set_layers(self):
        # array of fully connected layers
        self._layers = nn.ModuleList()

        # Initialise the input layer when it is requested (having a positive size hidden layer).
        if self.list_hidden_sizes[0] > 0:
            self._layers.append(
                self.module_linearity(
                    self.input_size, self.list_hidden_sizes[0], self.biases[0]
                )
            )
        # initialise the hidden layers
        for i in range(len(self.list_hidden_sizes) - 1):
            self._layers.append(
                self.module_linearity(
                    self.list_hidden_sizes[i],
                    self.list_hidden_sizes[i + 1],
                    self.biases[i + 1],
                )
            )

        # The output layer, which can be the only layer if no hidden layers (hidden_sizes = [0]) are given.
        input_size_for_output = (
            self.input_size
            if self.list_hidden_sizes[-1] == 0
            else self.list_hidden_sizes[-1]
        )
        self._layers.append(
            self.module_linearity(
                input_size_for_output, self.output_size, self.biases[-1]
            )
        )

        self._apply_dropout = nn.Dropout(p=self.dropout)

        # init the weights in the xavier way.
        self.apply(self.init_weights)

    def forward(self, x):
        # pass through the input layer, no dropout by definition.

        ##### For debugging purposes, turn this on with the init in basic_nnexp and check whether the parametric exp model has optimal loss.
        # self._layers[0].bias.data[0] = -1E10
        if len(self.activation_functions):
            x = self.activation_functions[0](self._layers[0](x))

        # Pass through the hidden layers.
        for layer_index in range(1, len(self.list_hidden_sizes)):
            x = self.activation_functions[layer_index](
                self._apply_dropout(self._layers[layer_index](x))
            )

        x = self._layers[-1](x)
        return x

    @staticmethod
    def init_weights(layer):
        if type(layer) == nn.Linear and layer.weight.requires_grad:
            gain = nn.init.calculate_gain("sigmoid")
            torch.nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None and layer.bias.requires_grad:
                layer.bias.data.fill_(0)
        return

    # section ######################################################################
    #  #############################################################################
    #  SETTERS / GETTERS

    # WIP: actually all of that should be private! We do not want to modify the model's parameter after init...
    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, new_input_size):
        if isinstance(new_input_size, int):
            self._input_size = new_input_size
        else:
            raise Error_type_setter(f"Argument is not an {str(int)}.")

    @property
    def list_hidden_sizes(self):
        return self._list_hidden_sizes

    @list_hidden_sizes.setter
    def list_hidden_sizes(self, new_list_hidden_sizes):
        if function_iterable.is_iterable(new_list_hidden_sizes):
            self._list_hidden_sizes = new_list_hidden_sizes
        else:
            raise Error_type_setter(f"Argument is not an Iterable.")

    @property
    def output_size(self):
        return self._output_size

    @output_size.setter
    def output_size(self, new_output_size):
        if isinstance(new_output_size, int):
            self._output_size = new_output_size
        else:
            raise Error_type_setter(f"Argument is not an {str(int)}.")

    @property
    def biases(self):
        return self._biases

    # always set the biases after list_hidden_sizes:
    @biases.setter
    def biases(self, new_biases):
        if function_iterable.is_iterable(new_biases):
            assert len(new_biases) == len(self.list_hidden_sizes) + 1 or (
                    len(new_biases) == 1 and self.is_model_without_hidden_layers
            ), "Passed activation functions' length does not match the number of hidden sizes."
            self._biases = new_biases
        else:
            raise Error_type_setter(f"Argument is not an iterable.")

    @property
    def activation_functions(self):
        return self._activation_functions

    @activation_functions.setter
    def activation_functions(self, new_activation_functions):
        if function_iterable.is_iterable(new_activation_functions):
            assert len(new_activation_functions) == len(self.list_hidden_sizes) or (
                    not len(new_activation_functions)
                    and self.is_model_without_hidden_layers
            ), "Passed activation functions do not the number of hidden sizes."
            self._activation_functions = new_activation_functions
        else:
            raise Error_type_setter(f"Argument is not an iterable.")

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, new_dropout):
        # Dropout is a float between 0 and 1.
        if isinstance(new_dropout, float) and 0 <= new_dropout < 1:
            self._dropout = new_dropout
        else:
            raise Error_type_setter(
                f"Argument is not a {str(float)} between 0. and 1.."
            )
