## copied from PhD-ASP-Methods/NL/projector_isa.py
## copied Dec 4 2024

from types import SimpleNamespace
from typing import Optional, Literal
from copy import deepcopy
import numpy as np
import pandas as pd
import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import keras
#import tensorflow as tf
from scipy.spatial import distance
import keras_tuner as kt
from shutil import rmtree

import multiprocessing as mp

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed 2) backend random seed  3) `python` random seed

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
# tf.config.experimental.enable_op_determinism()


class Feedforward:
    def __init__(self, num_hidden_layers: int = 3, num_hidden_units: int = 64,
                 activation: str = 'relu', use_bias: bool = True):
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units
        self.activation = activation
        self.use_bias = use_bias

    def get(self, input_length: int, output_length: int):
        network = keras.Sequential()
        network.add(keras.Input(shape=(input_length, )))
        for _ in range(self.num_hidden_layers):
            network.add(
                keras.layers.Dense(units=self.num_hidden_units, activation=self.activation, use_bias=self.use_bias)
            )
        network.add(keras.layers.Dense(units=output_length, use_bias=self.use_bias))
        network(keras.Input(shape=(input_length, )))
        return network


class Projector:
    """
    A class to fit a linear or nonlinear projection to instance space, given a preprocessed feature matrix and preprocessed performance matrix.

    Parameters:

        mode: :class:`Literal['linear', 'nonlinear'],`, **default=''linear''** --
            The primary parameter is whether linear or nonlinear models are used. If linear, no biases are used by default.

        instance_encoder_num_hidden_layers: :class:`Optional[int]`, **default=None** --
            The number of hidden layers in the instance encoder. Default depends on mode (0 for linear, 3 for nonlinear).

        instance_encoder_num_hidden_units: :class:`int`, **default=64** --
            The number of hidden units in each layer of the instance encoder

        instance_encoder_activation: :class:`str`, **default='relu'** --
            The activation function used in the instance encoder

        instance_decoder_num_hidden_layers: :class:`Optional[int]`, **default=None** --
            The number of hidden layers in the instance decoder. Default depends on mode (0 for linear, 3 for nonlinear).

        instance_decoder_num_hidden_units: :class:`int`, **default=64** --
            The number of hidden units in each layer of the instance decoder

        instance_decoder_activation: :class:`str`, **default='relu'** --
            The activation function used in the instance decoder

        performance_num_hidden_layers: :class:`Optional[int]`, **default=None** --
            The number of hidden layers in the performance model. Default depends on mode (0 for linear, 3 for nonlinear).

        performance_num_hidden_units: :class:`int`, **default=64** --
            The number of hidden units in each layer of the performance model.

        performance_activation: :class:`str`, **default='relu'** --
            The activation function used in the performance model.

        optimizer: :class:`str`, **default='Adam'** --
            The optimisation algorithm for training the model

        use_bias: :class:`Optional[bool]`, **default=None** --
            Whether to use a bias for network layers. If mode=linear, default is use_bias=False. If mode=nonlinear, True.

        feature_loss_weight: :class:`float`, **default=1.0** --
            The weight of the feature loss in the loss function

        performance_loss_weight: :class:`float`, **default=1.0** --
            The weight of the performance loss in the loss function.

    """
    def __init__(
            self,
            mode: Literal['linear', 'nonlinear'] = 'linear',
            instance_encoder_num_hidden_layers: Optional[int] = None,
            instance_encoder_num_hidden_units: int = 64,
            instance_encoder_activation: str = 'relu',
            instance_decoder_num_hidden_layers: Optional[int] = None,
            instance_decoder_num_hidden_units: int = 64,
            instance_decoder_activation: str = 'relu',
            performance_num_hidden_layers: Optional[int] = None,
            performance_num_hidden_units: int = 64,
            performance_activation: str = 'relu',
            optimizer: str = "Adam",
            use_bias: Optional[bool] = None,
            feature_loss_weight: float = 1.0,
            performance_loss_weight: float = 1.0,
            seed: int = 0
    ):
        self.mode = mode
        self.instance_encoder_num_hidden_layers = instance_encoder_num_hidden_layers
        self.instance_encoder_num_hidden_units = instance_encoder_num_hidden_units
        self.instance_encoder_activation = instance_encoder_activation
        self.instance_decoder_num_hidden_layers = instance_decoder_num_hidden_layers
        self.instance_decoder_num_hidden_units = instance_decoder_num_hidden_units
        self.instance_decoder_activation = instance_decoder_activation
        self.performance_num_hidden_layers = performance_num_hidden_layers
        self.performance_num_hidden_units = performance_num_hidden_units
        self.performance_activation = performance_activation
        self.use_bias = use_bias
        self.feature_loss_weight = feature_loss_weight
        self.performance_loss_weight = performance_loss_weight
        self.optimizer = optimizer

        keras.utils.set_random_seed(seed)


    def fit(self, feature_matrix: np.ndarray, algorithm_matrix: np.ndarray,
            batch_size: int = 32,
            epochs: int = 100,
            validation_split: float = 0.1,
            dim: int = 2
            ):
        """
        Fit method to train the projection model on the given feature_matrix and algorithm_matrix (both assumed to be preprocessed).

        :param feature_matrix: The preprocessed feature matrix
        :param algorithm_matrix: The preprocessed algorithm performance matrix
        :param batch_size: The batch size for training (default = 32)
        :param epochs: The number of epochs for training (default = 100)
        :param validation_split: The validation split ratio (default = 0.1)

        Returns:

        A SimpleNamespace with the following attributes:
            - z: a numpy array of the 2D instance space coordinates
            - model: The compiled Keras model
            - instance_encoder: The instance_encoder model
            - instance_encoder_weights: The weights of the instance_encoder model
            - instance_decoder: The instance_decoder model
            - instance_decoder_weights: The weights of the instance_decoder model
            - performance_model: The performance_model
            - performance_model_weights: The weights of the performance_model
            - history: The training history of the model
        """

        # default setting for bias depends on mode
        if self.use_bias is None:
            if self.mode == 'linear':
                self.use_bias = False
            if self.mode == 'nonlinear':
                self.use_bias = True

        # network layer defaults depends on mode
        if self.mode == 'linear':
            if self.instance_encoder_num_hidden_layers is None:
                self.instance_encoder_num_hidden_layers = 0
            if self.instance_decoder_num_hidden_layers is None:
                self.instance_decoder_num_hidden_layers = 0
            if self.performance_num_hidden_layers is None:
                self.performance_num_hidden_layers = 0

        if self.mode == 'nonlinear':
            if self.instance_encoder_num_hidden_layers is None:
                self.instance_encoder_num_hidden_layers = 3
            if self.instance_decoder_num_hidden_layers is None:
                self.instance_decoder_num_hidden_layers = 3
            if self.performance_num_hidden_layers is None:
                self.performance_num_hidden_layers = 3

        # instance encoder model
        num_features = feature_matrix.shape[1]
        instance_encoder = Feedforward(
            num_hidden_layers=self.instance_encoder_num_hidden_layers,
            num_hidden_units=self.instance_encoder_num_hidden_units,
            activation=self.instance_encoder_activation,
            use_bias=self.use_bias
        ).get(input_length=num_features, output_length=dim)
        instance_encoder_input = instance_encoder.input
        instance_encoder_output = instance_encoder(instance_encoder_input)

        # instance decoder model
        instance_decoder = Feedforward(
            num_hidden_layers=self.instance_decoder_num_hidden_layers,
            num_hidden_units=self.instance_decoder_num_hidden_units,
            activation=self.instance_decoder_activation,
            use_bias=self.use_bias
        ).get(input_length=dim, output_length=num_features)
        instance_decoder_input = instance_encoder_output
        instance_decoder_output = instance_decoder(instance_decoder_input)
        instance_decoder._name = 'feature'

        # performance model
        num_algos = algorithm_matrix.shape[1]
        performance_model = Feedforward(
            num_hidden_layers=self.performance_num_hidden_layers,
            num_hidden_units=self.performance_num_hidden_units,
            activation=self.performance_activation,
            use_bias=self.use_bias
        ).get(input_length=dim, output_length=num_algos)
        performance_model_output = performance_model(instance_encoder_output)
        performance_model._name = 'performance'

        # fit the model
        model = keras.Model(
            inputs=instance_encoder_input,
            outputs=[instance_decoder_output, performance_model_output]
        )
        model.compile(
            optimizer=self.optimizer,
            loss=('mse', 'mse'),
            loss_weights=(self.feature_loss_weight, self.performance_loss_weight)
        )

        # callback to stop overfitting
        # early_stopping = keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=5, 
        #     verbose=0, restore_best_weights=True
        # )

        history = model.fit(
            feature_matrix, [feature_matrix, algorithm_matrix],
            batch_size=batch_size, epochs=epochs, validation_split=validation_split,
            verbose=0, #callbacks=[early_stopping]
        )

        z = instance_encoder.predict(feature_matrix, verbose=0)

        ## save hyperparameters
        hyperparameters = {
            # 'mode': self.mode,
            'instance_encoder_num_hidden_layers': self.instance_encoder_num_hidden_layers,
            'instance_encoder_activation': self.instance_encoder_activation,
            'instance_decoder_num_hidden_layers': self.instance_decoder_num_hidden_layers,
            'instance_decoder_activation': self.instance_decoder_activation,
            'performance_num_hidden_layers': self.performance_num_hidden_layers,
            'performance_activation': self.performance_activation,
            'epochs': epochs,
            # 'optimizer': self.optimizer,
            # 'use_bias': self.use_bias
        }

        return SimpleNamespace(
            z=z,
            model=model,
            instance_encoder=instance_encoder,
            #instance_encoder_weights=instance_encoder.weights,
            instance_decoder=instance_decoder,
            #instance_decoder_weights=instance_decoder.weights,
            performance_model=performance_model,
            #performance_model_weights=performance_model.weights,
            #history=history,
            hyperparameters=hyperparameters
        )

    def tune_nl(self, feature_matrix: np.ndarray, algorithm_matrix: np.ndarray,
            max_trials: int,
            batch_size: int = 32,
            epochs: int = 100,
            validation_split: float = 0.1,
            dim: int = 2,
            outpath = ''
            ):
        # only run for nonlinear mode
        if self.mode != 'nonlinear':
            return
        
        if self.use_bias is None:
            self.use_bias = True
        
        num_features = feature_matrix.shape[1]
        num_algos = algorithm_matrix.shape[1]            
            
        def model_builder(hp):
            # instance encoder model
            instance_activation = hp.Choice('instance_activation',
                                values=['relu', 'tanh', 'sigmoid'], default='relu')
            instance_num_hidden_layers = hp.Int('instance_num_hidden_layers',
                                min_value=1, max_value=5, default=3)
            instance_encoder = Feedforward(
                num_hidden_layers=instance_num_hidden_layers,
                num_hidden_units=self.instance_encoder_num_hidden_units,
                activation=instance_activation,
                use_bias=self.use_bias
            ).get(input_length=num_features, output_length=dim)
            instance_encoder_input = instance_encoder.input
            instance_encoder_output = instance_encoder(instance_encoder_input)

            # instance decoder model
            instance_decoder = Feedforward(
                num_hidden_layers=instance_num_hidden_layers,
                num_hidden_units=self.instance_decoder_num_hidden_units,
                activation=instance_activation,
                use_bias=self.use_bias
            ).get(input_length=dim, output_length=num_features)
            instance_decoder_input = instance_encoder_output
            instance_decoder_output = instance_decoder(instance_decoder_input)
            instance_decoder._name = 'feature'

            # performance model
            performance_model = Feedforward(
                num_hidden_layers=hp.Int('performance_num_hidden_layers',
                                min_value=1, max_value=5, default=3),
                num_hidden_units=self.performance_num_hidden_units,
                activation=hp.Choice('performance_activation',
                                values=['relu', 'tanh', 'sigmoid'], default='relu'),
                use_bias=self.use_bias
            ).get(input_length=dim, output_length=num_algos)
            performance_model_output = performance_model(instance_encoder_output)
            performance_model._name = 'performance'

            # fit the model
            model = keras.Model(
                inputs=instance_encoder_input,
                outputs=[instance_decoder_output, performance_model_output]
            )
            model.compile(
                optimizer=self.optimizer,
                loss=('mse', 'mse'),
                loss_weights=(self.feature_loss_weight, self.performance_loss_weight)
            )

            return model

        tic = time.time()
        if max_trials == 0:
            tuner = kt.Hyperband(model_builder, objective='val_loss',max_epochs=epochs,seed=111,
                                overwrite=True,directory=f'{outpath}hp')
        else:
            tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials=max_trials, seed=111, 
                                    overwrite=True, directory=f'{outpath}hp')

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=1, 
            verbose=0, restore_best_weights=True)
        
        tuner.search(feature_matrix, [feature_matrix, algorithm_matrix], 
                     batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                     callbacks=[early_stopping], verbose=0)
        
        print('AE tuner time:', time.time()-tic)
        # delete the directory
        rmtree(f'{outpath}hp')
        
        # set the best hyperparameters
        best_hps = tuner.get_best_hyperparameters()[0].values
        self.instance_encoder_num_hidden_layers = best_hps['instance_num_hidden_layers']
        self.instance_encoder_activation = best_hps['instance_activation']
        self.instance_decoder_num_hidden_layers = best_hps['instance_num_hidden_layers']
        self.instance_decoder_activation = best_hps['instance_activation']
        self.performance_num_hidden_layers = best_hps['performance_num_hidden_layers']
        self.performance_activation = best_hps['performance_activation']
        
        return best_hps

    
    def _run(self, feature_matrix, algorithm_matrix, batch_size, epochs, validation_split, dim, D_high):
        projMod = self.fit(feature_matrix, algorithm_matrix, 
                            batch_size, epochs, validation_split, dim)
        
        f_i = self.error(feature_matrix, algorithm_matrix, projMod.model)['loss']

        z = projMod.z
        D_low = distance.pdist(z, 'euclidean')
        rho = np.corrcoef(D_high, D_low)[0,1]

        return {'rho': rho, 
                'f': f_i, 
                'model': deepcopy(projMod)}

    def multi_fit(self, feature_matrix: np.ndarray, algorithm_matrix: np.ndarray,
        max_tries:int = 5,batch_size: int = 32, epochs: int = 100, validation_split: float = 0.1,
        dim: int = 2, n_cores: int = 1):

        D_high = distance.pdist(feature_matrix, 'euclidean')

        if n_cores < 1:
            pool = mp.Pool(n_cores)
            results = [pool.apply_async(self._run, args=(feature_matrix, algorithm_matrix, batch_size, epochs, validation_split, dim, D_high)) for _ in range(max_tries)]

            pool.close()
            pool.join()

            sols = [res.get() for res in results]
            
        else:
            sols = [self._run(feature_matrix, algorithm_matrix, batch_size, epochs, validation_split, dim, D_high) for _ in range(max_tries)]
        
        best_sol = max(sols, key=lambda x: x['rho'])
        # best_sol = min(sols, key=lambda x: x['f'])

        return best_sol['model']
    

    def multi_fit0(self, feature_matrix: np.ndarray, algorithm_matrix: np.ndarray,
        tries = 3, max_tries = 0,batch_size: int = 32, epochs: int = 100, validation_split: float = 0.1,
        dim: int = 2):

        if max_tries == 0:
            max_tries = tries*5
        t = 0

        sol = {'rho':-np.inf, 'f':np.inf}
        bf = {'rho':-np.inf, 'f':np.inf}

        ## TOD0 - allow different choices for D_high
        D_high = distance.pdist(feature_matrix, 'euclidean')

        
        for i in range(tries):
            print(f'Autoencoder solution try {i+1}', end='\r')

            t += 1
            projMod = self.fit(feature_matrix, algorithm_matrix, 
                             batch_size, epochs, validation_split, dim)
            
            f_i = self.error(feature_matrix, algorithm_matrix, projMod.model)['loss']

            z = projMod.z
            D_low = distance.pdist(z, 'euclidean')
            rho = np.corrcoef(D_high, D_low)[0,1]

            
            if rho > max(bf['rho'],sol['rho']) or f_i < min(bf['f'],sol['f']):
                sol['rho'] = rho
                sol['f'] = f_i
                sol['z'] = z
                sol['model'] = deepcopy(projMod)
                # print(f'{i} - rho: {rho} - f: {f_i}')

            bf['rho'] = max(bf['rho'],rho)
            bf['f'] = min(bf['f'],f_i)

            
            # while t>1 and  f_i < sol['f'] and t < max_tries:
            #     print(f'Solution has not converged yet. Trying again. ({t} - {f_i})')
            #     t += 1
            #     sol['rho'] = np.inf

            #     projMod = self.fit(feature_matrix, algorithm_matrix, 
            #                  batch_size, epochs, validation_split, dim)
            
            #     f_i = self.error(feature_matrix, algorithm_matrix, projMod.model)['loss']

            

        return sol['model']
    

    @classmethod
    def error(self, feature_matrix: np.ndarray, algorithm_matrix: np.ndarray, model: keras.Model):

        """
        Calculate the reconstruction error for the given feature_matrix and algorithm_matrix.

        :param feature_matrix: The preprocessed feature matrix
        :param algorithm_matrix: The preprocessed algorithm performance matrix
        :param model: The trained model

        Returns:

        
        """
        errors = model.evaluate(feature_matrix, [feature_matrix, algorithm_matrix], 
                       verbose=0)
        
        return dict(zip(['loss','F loss','Y loss'], errors))


if __name__ == '__main__':
    feature_data = pd.read_csv('./NL/feature_process.csv')
    feature_matrix = feature_data.iloc[:, 1:].to_numpy()
    algorithm_data = pd.read_csv('./NL/algorithm_process.csv')
    algorithm_matrix = algorithm_data.iloc[:, 1:].to_numpy() 

    proj = Projector(mode='nonlinear', seed=111)
    # proj_results = proj.multi_fit(feature_matrix, algorithm_matrix,
    #                           max_tries=7,validation_split=0,n_cores=3)

    proj_params = proj.tune_nl(feature_matrix, algorithm_matrix, epochs=10,
                                    max_trials=0, validation_split=0.1)