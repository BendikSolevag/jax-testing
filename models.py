import random
import jax.nn
import jax.numpy as jnp
random.seed(3)

class Model:    
    def get_params(self):
        return self.params
    def set_params(self, params):
        self.params = params
    def generate_params():
        return AssertionError("Not implemented")
    def forward():
        return AssertionError("Not implemented")
    def update_params():
        return AssertionError("Not implemented")


class PID(Model):
    def generate_params(self, initial_params_max, initial_params_min):
        Kp = random.uniform(initial_params_min, initial_params_max)
        Ki = random.uniform(initial_params_min, initial_params_max)
        Kd = random.uniform(initial_params_min, initial_params_max)
        self.params = [jnp.array([Kp, Ki, Kd])]

    def forward(self, errors, t, params):
        """Forward pass of the PID controller"""
        e = errors[t-1] if t > 0 else 0
        e_sum = jnp.sum(errors)
        de = errors[t-1] - errors[t-2] if t > 1 else 0
        x = jnp.array([e, e_sum, de])
        for layer in params:
            x = jnp.dot(x, layer)
        return x
    
    def update_params(self, params, grads, scale):
        updated_params = []
        for layer, grad in zip(params, grads):
            layer = layer - grad * scale
            updated_params.append(layer)
        self.params = updated_params



class NN(Model):
    def generate_params(self, initial_params_min: float, initial_params_max: float, layer_sizes: list[int], input_size: int, activation: str):
        params = []
        for nnodes in layer_sizes:
            weights = [[ random.uniform(initial_params_min, initial_params_max) for _ in range(nnodes)] for _ in range(input_size)]
            biases = [random.uniform(initial_params_min, initial_params_max) for _ in range(nnodes)]
            layer = [jnp.array(weights), jnp.array(biases)]
            params.append(layer)
            input_size = nnodes
        self.params = params

        if activation == "tanh":
            self.activation = jax.nn.tanh
        elif activation == "sigmoid":
            self.activation = jax.nn.sigmoid
        elif activation == "relu":
            self.activation = jax.nn.relu
        else:
            self.activation = lambda x: x

    def forward(self, errors, t, params):
        """Forward pass of the NN controller"""
        e = errors[t-1] if t > 0 else 0
        de = errors[t-1] - errors[t-2] if t > 1 else 0
        sum_e = jnp.sum(errors)
        x = jnp.array([e, de, sum_e])
        for i, [w, b] in enumerate(params):
            x = jnp.dot(x, w) + b
            if i < len(params) - 1:
                x = self.activation(x)
        return x[0]
    
    def update_params(self, params, grads, scale):
        updated_params = []
        for [weights, biases], [d_weights, d_biases] in zip(params, grads):
            weights = weights - d_weights * scale
            biases = biases - d_biases * scale
            updated_params.append([weights, biases])
        self.params = updated_params


