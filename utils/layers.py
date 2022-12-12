import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np

from utils.circuits import generate_cp_pg_circuit


class TrainableRescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim, trainable_obs_weight):
        super(TrainableRescaling, self).__init__()
        self.input_dim = input_dim
        self.trainable_obs_weight = trainable_obs_weight
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1, input_dim)), dtype=tf.dtypes.float32,
            trainable=self.trainable_obs_weight, name="trainable_rescaling")

    def call(self, inputs):
        return tf.math.multiply(
            inputs,
            tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0))


class EquivariantLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_input_params,
            n_vars,
            n_edges,
            circuit_depth,
            params,
            param_perturbation=0,
            name="equivariant_layer"):
        super(EquivariantLayer, self).__init__(name=name)
        self.num_input_params = num_input_params * circuit_depth
        self.num_params = 2 * circuit_depth
        self.circuit_depth = circuit_depth
        self.param_perturbation = param_perturbation

        param_init = tf.ones(shape=(1, self.num_params), dtype=tf.dtypes.float32)
        self.params = tf.Variable(
            initial_value=param_init,
            trainable=True, name="params"
        )

        self.param_repeats = []
        for layer in range(self.circuit_depth):
            self.param_repeats.append(n_vars)
            self.param_repeats.append(n_edges)

        alphabetical_params = sorted(params)
        self.indices = tf.constant([params.index(a) for a in alphabetical_params])

    def call(self, inputs):
        repeated_params = tf.repeat(self.params, repeats=self.param_repeats)

        repeat_inputs = tf.reshape(
            tf.repeat(inputs, repeats=self.circuit_depth, axis=0),
            [tf.shape(inputs)[0], self.num_input_params])

        if self.param_perturbation > 0:
            perturbation_vec = tf.random.normal(
                shape=tf.shape(repeated_params),
                mean=0, stddev=self.param_perturbation)
            perturbed_params = tf.math.add(repeated_params, perturbation_vec)
            repeated_params = perturbed_params

        data_values = tf.math.multiply(repeat_inputs, repeated_params)
        output = tf.gather(data_values, self.indices, axis=1)

        return output


class SoftmaxInverseTemp(tf.keras.layers.Layer):
    def __init__(self, trainable_beta, beta_init=1, n_shots=0):
        super(SoftmaxInverseTemp, self).__init__()
        self.n_shots = n_shots
        self.beta = tf.Variable(
            initial_value=tf.ones(shape=1, dtype=tf.dtypes.float32) * beta_init,
            trainable=trainable_beta, name="output_params")
        self.softmax_layer = tf.keras.layers.Softmax()

    def call(self, inputs):
        if self.n_shots > 0:
            shot_noise = tf.random.normal(tf.shape(inputs)) * 1/np.sqrt(self.n_shots)
            inputs = inputs + shot_noise

        return self.softmax_layer(
            tf.math.multiply(
                inputs, self.beta))


class ReUploadingPQC(tf.keras.layers.Layer):
    def __init__(
            self,
            qubits,
            n_layers,
            observables,
            noise_p=0.,
            param_perturbation=0.0,
            n_shots=0,
            activation="linear",
            name="re-uploading_PQC"):

        super(ReUploadingPQC, self).__init__(name=name)

        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.noise_p = noise_p
        self.param_perturbation = param_perturbation

        circuit, theta_symbols, input_symbols = generate_cp_pg_circuit(
            qubits, n_layers, noise_p=self.noise_p)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

        if self.noise_p >= 1e-5:
            self.computation_layer = tfq.layers.NoisyControlledPQC(circuit, observables,
                                                                   sample_based=False, repetitions=10)
        else:
            if n_shots > 0:
                self.computation_layer = tfq.layers.ControlledPQC(
                    circuit, observables, differentiator=tfq.differentiators.ParameterShift(),
                    repetitions=n_shots)
            else:
                self.computation_layer = tfq.layers.ControlledPQC(
                    circuit, observables, differentiator=tfq.differentiators.Adjoint())

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        if self.param_perturbation > 0:
            perturbation_vec = tf.random.normal(
                shape=tf.shape(joined_vars),
                mean=0, stddev=self.param_perturbation)
            perturbed_vars = tf.math.add(joined_vars, perturbation_vec)
            joined_vars = perturbed_vars

        return self.computation_layer([tiled_up_circuits, joined_vars])


class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.) ** i for i in range(output_dim)]]), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class ScalableDataReuploadingController(tf.keras.layers.Layer):
    def __init__(
            self,
            num_input_params,
            num_params,
            circuit_depth,
            params,
            trainable_scaling=True,
            use_reuploading=True,
            param_perturbation=0,
            name="scalable_data_reuploading"):
        super(ScalableDataReuploadingController, self).__init__(name=name)
        self.num_params = num_params
        self.param_perturbation = param_perturbation
        self.circuit_depth = circuit_depth
        self.use_reuploading = use_reuploading
        self.num_input_params = num_input_params
        if self.use_reuploading:
            self.num_input_params *= circuit_depth

        param_init = tf.random_uniform_initializer(minval=0., maxval=np.pi)
        self.params = tf.Variable(
            initial_value=param_init(shape=(1, num_params), dtype=tf.dtypes.float32),
            trainable=True, name="params"
        )

        input_param_init = tf.ones(shape=(1, self.num_input_params))
        self.input_params = tf.Variable(
            initial_value=input_param_init, dtype=tf.dtypes.float32,
            trainable=trainable_scaling, name="input_params"
        )

        alphabetical_params = sorted(params)
        self.indices = tf.constant([alphabetical_params.index(a) for a in params])

    def call(self, inputs):
        output = tf.repeat(self.params, repeats=tf.shape(inputs)[0], axis=0)

        input_repeats = self.circuit_depth if self.use_reuploading else 1
        repeat_inputs = tf.repeat(inputs, repeats=input_repeats, axis=1)
        repeat_inp_weights = tf.repeat(self.input_params, repeats=tf.shape(inputs)[0], axis=0)

        output = tf.concat(
            [
                output,
                tf.keras.layers.Activation('tanh')(tf.math.multiply(repeat_inputs, repeat_inp_weights))
            ], 1)

        output = tf.gather(output, self.indices, axis=1)

        if self.param_perturbation > 0:
            perturbation_vec = tf.random.normal(
                shape=tf.shape(output),
                mean=0, stddev=self.param_perturbation)
            perturbed_vars = tf.math.add(output, perturbation_vec)
            output = perturbed_vars

        return output