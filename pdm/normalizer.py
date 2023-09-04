from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Optimizer, Adamax
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import MeanSquaredError, Loss
from tensorflow import Tensor
from typing import Union


class DescriptorPredictor(Model):


    def __init__(self,
                 output_size: int = 14,
                 n_dense: int = 5,
                 neurons: Union[list[int], int] = 64,
                 callbacks: list[callable] = [],
                 optimizer: Optimizer = Adamax(),
                 loss_fn: Loss = MeanSquaredError(),
                 l2_regularization: float = 0.005):
        """Model initializer

        Parameters
        ----------
        output_size : int, optional
            Number of sensor variables to predict, by default 14
        n_dense : int, optional
            Number of dense (fnn) layers in the model, by default 5
        neurons : Union[list[int], int], optional
            Number of neurons to be used in FNN, can be represented as a list, by default 64
        callbacks : list[callable], optional
            Tensorflow Keras callbacks to be later used in training, by default []
        optimizer : Optimizer, optional
            Tensorflow Keras Optimizer, by default Adamax()
        loss_fn : Loss, optional
            Loss function for computing gradients, by default MeanSquaredError()
        l2_regularization : float, optional
            L2 regularization for layer weights, by default 0.005

        Raises
        ------
        ValueError
            In case the provided list of neurons doesn't match the number
            of deep layers specified in the init method
        """


        super(DescriptorPredictor, self).__init__()
        self.out = Dense(output_size, name='Output')
        self.n_dense = n_dense
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.l2_regularization = l2_regularization

        # Creates a block of Dense layers
        if isinstance(neurons, int):
           self.neurons = [neurons] * n_dense
        elif isinstance(neurons, list):
            if len(neurons) != len(self.n_dense):
                raise ValueError(f"Number of neurons {neurons} doesn't match number of dense layers {self.n_dense}")
            else:
                self.neurons = neurons
        self.dense_layers = []

        for i, neuron in zip(range(self.n_dense), self.neurons):
            self.dense_layers.append(
                Dense(neuron,
                      activation='relu',
                      kernel_regularizer=L2(self.l2_regularization),
                      name=f'Dense_{i}')
            )


    def call(self, input_tensor: Tensor) -> Tensor:
        """Forward method of the model

        Parameters
        ----------
        input_tensor : Tensor
            Input of the model

        Returns
        -------
        Tensor
            Output of the model
        """        

        x = input_tensor
        for layer in self.dense_layers:
            x = layer(x)
        x = self.out(x)

        return x
