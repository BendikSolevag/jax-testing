config = {    
    "plant": 'cournot', # bathtub, cournot, car
    "controller": 'neural', # pid, neural
    "layer_sizes": [3,3,1],
    "activation": 'relu', # sigmoid, tanh, relu, none
    "initial_params_max": -1,
    "initial_params_min": 1,
    "max_timesteps": 250,
    "epochs": 50,
    "disturbance_max": 0.1,
    "disturbance_min": -0.1,
    "learning_rate": 0.01,
    "H_init": 100,
    "A": 10,
    "C": 0.1,

    "T": 10,
    "p_max": 10,
    "cm": 0.1,

    "desired_speed": 100,
    "ground_friction_coefficient": 0.01,
    "car_mass": 1, 
}