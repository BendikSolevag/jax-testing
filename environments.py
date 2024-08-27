import jax.numpy as jnp

class Environment:
    def loss():
        return AssertionError("Not implemented")
    def forward():
        return AssertionError("Not implemented")
    def update_state_variables():
        return AssertionError("Not implemented")
    def get_initial_state_variable(self):
        return AssertionError("Not implemented")

class Tub(Environment):
    def __init__(self, H_init, A, C):
        self.H_init = H_init
        self.A = A
        self.C = C

    def get_initial_state_variable(self):
        return self.H_init

    def loss(self, H):
        return jnp.abs(self.H_init - H)
    
    def update_state_variables(self, U, D):
        # No state variables to update
        return

    def forward(self, U, D, H):
        V = jnp.sqrt(2*9.81*H) if H > 0 else 0.0
        Q = V * self.C
        B_new = (self.A * H) + U + D - Q 
        return B_new / self.A


class Cournot(Environment):
    def __init__(self, T, cm, p_max):
        self.T = T
        self.cm = cm
        self.p_max = p_max
        self.q_1 = 3
        self.q_2 = 3

    
    def get_initial_state_variable(self):
        self.q_1 = 3
        self.q_2 = 3
        return self.p_max - self.q_1 - self.q_2

    def loss(self, P, q):
        return jnp.abs(self.T - (q * (P - self.cm)))

    def update_state_variables(self, U, D):
        self.q_1 = self.q_1 + U
        self.q_2 = max(self.q_2 + D, 0)
    
    def forward(self, q_1):
        p = self.p_max - q_1 - self.q_2
        return p


class Car(Environment):
    def __init__(self, desired_speed, ground_friction_coefficient, mass):
        self.desired_speed = desired_speed
        self.ground_friction_coefficient = ground_friction_coefficient
        self.mass = mass

    def get_initial_state_variable(self):
        return 1
    
    def update_state_variables(self, U, D):
        # No state variables to update
        return

    def loss(self, speed):
        return jnp.abs(self.desired_speed - speed)
        
    def forward(self, U, D, speed):
        ground_friction = self.ground_friction_coefficient * self.mass * 9.81 * speed
        
        sigma_f = U + D - ground_friction
        
        a = sigma_f / self.mass
        
        # For simplicity's sake, 1 timestep = 1 second => m/s^2 * s = m/s
        return speed + a * 1
            