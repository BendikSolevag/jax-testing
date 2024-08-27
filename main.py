import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import Model, PID, NN
from environments import Car, Environment, Tub, Cournot
from config import config
randomkey = jax.random.PRNGKey(45)
from sys import exit
import datetime

class Consys:
    def __init__(self, model: Model, environment: Environment):
        self.model: Model = model
        self.environment: Environment = environment
        self.max_timesteps = config["max_timesteps"]
        self.epochs = config["epochs"]
        self.disturbance_min = config["disturbance_min"]
        self.disturbance_max = config["disturbance_max"]

    def epoch(self, progress_bar: tqdm, epoch: int):
        state_history = []
        output_history = []
        state_var = self.environment.get_initial_state_variable()
        errors = jnp.zeros((self.max_timesteps,))
        D = jax.random.uniform(randomkey,(self.max_timesteps,), minval=self.disturbance_min, maxval=self.disturbance_max)
        
        tmax = (epoch+1)*10
        timesteps = range(self.max_timesteps) if config["controller"] == 'neural' else range(tmax)
        for t in timesteps:
            progress_bar.update(1)
            # Handle model forward passs
            params = self.model.get_params()
            U, grads = jax.value_and_grad(self.model.forward, argnums=2)(errors, t, params)
            output_history.append(U)
            

            # Handle environment forward pass
            self.environment.update_state_variables(U, D[t])
            envfunc = jax.value_and_grad(self.environment.forward, argnums=0)
            environment_grad = None
            if config["plant"] in ['bathtub', 'car']:
                state_var, environment_grad = envfunc(U, D[t], state_var)
            elif config["plant"] == 'cournot':
                state_var, environment_grad = envfunc(self.environment.q_1)


            # Calculate Loss 
            lossfunc = jax.value_and_grad(self.environment.loss, argnums=0)
            err, loss_grad = None, None
            if config["plant"] in ['bathtub', 'car']:
                err, loss_grad = lossfunc(state_var)
            elif config["plant"] == 'cournot':
                err, loss_grad = lossfunc(state_var, self.environment.q_1)
            errors = errors.at[t].add(err)



            # Update model parameters 
            scale = environment_grad * loss_grad * config["learning_rate"]
            self.model.update_params(params, grads, scale)

            """print('t: ', t, 
                  'state_var: ', state_var, 
                  'U: ', U, 
                  'D: ', D[t], 
                  'err: ', err,
                  'scale: ', scale,
                  )"""
            

            # Misc 
            state_history.append(state_var)
        

        
        if config["controller"] == 'pid':
            #returned_error = jnp.mean(errors[:(epoch+1)*10])
            returned_error = errors[tmax - 1]
            return returned_error, self.model.get_params()

        #returned_error = jnp.mean(errors)
        returned_error = errors[-1]
        return returned_error
    
    def run(self):
        rets = None
        pbar = tqdm(total=self.epochs * self.max_timesteps)

        errors = []
        kp = []
        ki = []
        kd = []

        for i in range(config["epochs"]):
            if config["controller"] == 'pid':
                error, [rets] = self.epoch(pbar, i)
                errors.append(error)
                kp.append(rets[0].item())
                ki.append(rets[1].item())
                kd.append(rets[2].item())
            else:
                error = self.epoch(pbar, i)
                errors.append(error)
        

        now = datetime.datetime.now()
        plt.plot(errors)
        plt.title('Loss over time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MAE)')
        plt.savefig(f"{str(now)}_{config['plant']}_{config['controller']}.png", dpi=400)
        plt.close()

        if config["controller"] == 'pid':
            plt.plot(kp, label='P parameter')
            plt.plot(ki, label='I parameter')
            plt.plot(kd, label='D parameter')
            plt.title('PID parameters over time')
            plt.xlabel('Epochs')
            plt.ylabel('Parameter value')
            plt.savefig(f"{str(now)}_{config['plant']}_{config['controller']}_parameters.png", dpi=400)
            plt.close()

if __name__ == "__main__":
    model = None
    if config["controller"] == "pid":
        model = PID()
        model.generate_params(config["initial_params_min"], config["initial_params_max"])
    else:
        model = NN()
        initial_params_min = config["initial_params_min"]
        initial_params_max = config["initial_params_max"]
        activation = config["activation"]
        input_size = 3
        layer_sizes = config["layer_sizes"]
        model.generate_params(initial_params_min, initial_params_max, layer_sizes, input_size, activation)

    environment: Environment = None
    if config["plant"] == 'bathtub':
        environment = Tub(config["H_init"], config["A"], config["C"])
    elif config["plant"] == 'cournot':
        environment = Cournot(config["T"], config["cm"], config["p_max"])
    elif config["plant"] == 'car':
        desired_speed = config["desired_speed"]
        ground_friction_coefficient = config["ground_friction_coefficient"]
        car_mass = config["car_mass"]
        environment = Car(desired_speed, ground_friction_coefficient, car_mass)

    consys = Consys(model, environment)    
    consys.run()
