import sys
sys.path.append('./src/rl')

import time

import gym
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

# from gym_pybullet_drones.utils.Logger import Logger
from utils.Logger import Logger
from plotter import plot
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env

# define defaults
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

INIT_XYZS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(DEFAULT_NUM_DRONES)])
NUM_PHYSICS_STEPS = (
    int(DEFAULT_SIMULATION_FREQ_HZ / DEFAULT_CONTROL_FREQ_HZ)
    if DEFAULT_AGGREGATE
    else 1
)

PERIOD = 15
NUM_WP = DEFAULT_CONTROL_FREQ_HZ * PERIOD

# hyperparams for training
TRAIN_TRAJ_TYPE = "circle"
TEST_TRAJ_TYPE = "helix"
NUM_EPISODES = 1e3
ACTOR_NET_ARCH = [50, 50]
CRITIC_NET_ARCH = [50, 50]


def generate_target_trajectory(type="line", steps: int = NUM_WP):
    target_trajectory = np.zeros(shape=(steps, 3))

    if type == "hover":
        height = 0.3
        target_trajectory = (
            np.ones(shape=(steps, 3)) * np.array([0.0, 0.0, height]) + INIT_XYZS[0, 1]
        )

    elif type == "line":
        height = 0.3
        t = np.linspace(start=0.0, stop=height, num=steps)
        target_trajectory = np.transpose(np.vstack([t, t, t])) + INIT_XYZS[0, :]

    elif type == "circle":
        rad = 0.3
        freq = 1 / steps
        x = rad * np.cos(2 * np.pi * freq * np.arange(stop=steps))
        y = rad * np.sin(2 * np.pi * freq * np.arange(stop=steps))
        z = rad * np.ones_like(x)
        target_trajectory = np.transpose(np.vstack([x, y, z])) + INIT_RPYS[0, :]

    elif type == "helix":
        height = 0.3
        width = 0.3
        freq = 3 / steps
        z = np.linspace(start=0.0, stop=height, num=steps)
        x = width * np.cos(2 * np.pi * freq * np.arange(steps))
        y = width * np.sin(2 * np.pi * freq * np.arange(steps))

        target_trajectory = np.transpose(np.vstack([x, y, z])) + INIT_XYZS[0, :]

    return target_trajectory


def create_controller(pos_p, pos_i, pos_d, att_p, att_i, att_d):
    controller = DSLPIDControl(drone_model=DEFAULT_DRONES)

    controller.P_COEFF_FOR = pos_p
    controller.I_COEFF_FOR = pos_i
    controller.D_COEFF_FOR = pos_d
    controller.P_COEFF_TOR = att_p
    controller.I_COEFF_TOR = att_i
    controller.D_COEFF_TOR = att_d

    return controller


def simulate_trajectory(env, ctrl, trajectory, name="out.png"):
    logger = Logger(
        logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS),
        num_drones=1,
        output_folder=DEFAULT_OUTPUT_FOLDER,
    )

    # counter
    wp_counters = np.array(
        [int((i * NUM_WP / 6) % NUM_WP) for i in range(DEFAULT_NUM_DRONES)]
    )

    # control freq (in steps)
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / DEFAULT_CONTROL_FREQ_HZ))

    # map: drone_id - action
    action = {str(i): np.array([0, 0, 0, 0]) for i in range(DEFAULT_NUM_DRONES)}
    START = time.time()
    for i in range(0, int(DEFAULT_DURATION_SEC * env.SIM_FREQ), NUM_PHYSICS_STEPS):

        next_obs, reward, done, info = env.step(action)
        logger.log(
            drone=0,
            timestamp=i / env.SIM_FREQ,
            state=np.hstack(
                [
                    next_obs["0"]["state"][0:3],
                    next_obs["0"]["state"][10:13],
                    next_obs["0"]["state"][7:10],
                    next_obs["0"]["state"][13:20],
                    np.resize(action["0"], (4)),
                ]
            ),
            control=np.hstack(
                [trajectory[wp_counters[0], :], INIT_RPYS[0, :], np.zeros(6)]
            ),
        )

        # apply control
        if i % CTRL_EVERY_N_STEPS == 0:

            for j in range(DEFAULT_NUM_DRONES):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    state=next_obs[str(j)]["state"],
                    target_pos=np.hstack(trajectory[wp_counters[j], :]),
                    target_rpy=INIT_RPYS[j, :],
                )

            # increment control counter
            for j in range(DEFAULT_NUM_DRONES):
                wp_counters[j] = (
                    wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0
                )

        if i % env.SIM_FREQ == 0:
            env.render()

        if DEFAULT_GUI:
            sync(i, START, env.TIMESTEP)

    env.reset()

    logger.save_as_csv(comment=name)


def tune_pid_coeffs():
    # init empty
    tuned_P_coeff_for = np.empty(shape=(1, 3))
    tuned_I_coeff_for = np.empty(shape=(1, 3))
    tuned_D_coeff_for = np.empty(shape=(1, 3))
    tuned_P_coeff_tor = np.empty(shape=(1, 3))
    tuned_I_coeff_tor = np.empty(shape=(1, 3))
    tuned_D_coeff_tor = np.empty(shape=(1, 3))

    # create envs
    tune_env = gym.make("tune-aviary-v0")
    print("action space:", tune_env.action_space)
    print("observation space:", tune_env.observation_space)
    check_env(tune_env, warn=True, skip_render_check=True)

    # simulate
    ctrl_env = CtrlAviary(
        drone_model=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=DEFAULT_PHYSICS,
        neighbourhood_radius=10,
        freq=DEFAULT_SIMULATION_FREQ_HZ,
        aggregate_phy_steps=NUM_PHYSICS_STEPS,
        gui=DEFAULT_GUI,
        record=DEFAULT_RECORD_VIDEO,
        obstacles=DEFAULT_OBSTACLES,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    )

    # modify training trajectory
    # compute velocity as (s' - s) / dt
    steps = int(tune_env.SIM_FREQ * tune_env.EPISODE_LEN_SEC / tune_env.AGGR_PHY_STEPS)
    train_trajectory = generate_target_trajectory(type=TRAIN_TRAJ_TYPE, steps=steps)
    train_trajectory_vel = np.diff(train_trajectory) / tune_env.CTRL_TIMESTEP

    tune_env.TARGET_POSITION = train_trajectory
    tune_env.TARGET_VELOCITY = train_trajectory_vel

    # create test trajectory
    target_test_trajectory = generate_target_trajectory(
        type=TEST_TRAJ_TYPE, steps=NUM_WP
    )

    # create model for RL
    # multi-layered perceptron for actor and critic
    model = TD3(
        "MlpPolicy",
        tune_env,
        policy_kwargs=dict(net_arch=dict(pi=ACTOR_NET_ARCH, qf=CRITIC_NET_ARCH)),
    )

    print("-" * 10, "PID params before training")
    print(f"{tune_env.ctrl.P_COEFF_FOR=}")
    print(f"{tune_env.ctrl.I_COEFF_FOR=}")
    print(f"{tune_env.ctrl.D_COEFF_FOR=}")
    print(f"{tune_env.ctrl.P_COEFF_TOR=}")
    print(f"{tune_env.ctrl.I_COEFF_TOR=}")
    print(f"{tune_env.ctrl.D_COEFF_TOR=}")

    init_controller = create_controller(
        pos_p=tune_env.ctrl.P_COEFF_FOR,
        pos_i=tune_env.ctrl.I_COEFF_FOR,
        pos_d=tune_env.ctrl.D_COEFF_FOR,
        att_p=tune_env.ctrl.P_COEFF_TOR,
        att_i=tune_env.ctrl.I_COEFF_TOR,
        att_d=tune_env.ctrl.D_COEFF_TOR,
    )

    simulate_trajectory(
        ctrl_env, [init_controller], target_test_trajectory, name="init"
    )

    model.learn(total_timesteps=NUM_EPISODES)

    print("-" * 10, "PID params after training")
    print(f"{tune_env.ctrl.P_COEFF_FOR=}")
    print(f"{tune_env.ctrl.I_COEFF_FOR=}")
    print(f"{tune_env.ctrl.D_COEFF_FOR=}")
    print(f"{tune_env.ctrl.P_COEFF_TOR=}")
    print(f"{tune_env.ctrl.I_COEFF_TOR=}")
    print(f"{tune_env.ctrl.D_COEFF_TOR=}")

    tuned_P_coeff_for = tune_env.ctrl.P_COEFF_FOR
    tuned_I_coeff_for = tune_env.ctrl.I_COEFF_FOR
    tuned_D_coeff_for = tune_env.ctrl.D_COEFF_FOR
    tuned_P_coeff_tor = tune_env.ctrl.P_COEFF_TOR
    tuned_I_coeff_tor = tune_env.ctrl.I_COEFF_TOR
    tuned_D_coeff_tor = tune_env.ctrl.D_COEFF_TOR

    # tuned controller
    tuned_controller = DSLPIDControl(drone_model=DEFAULT_DRONES)

    tuned_controller.P_COEFF_FOR = tuned_P_coeff_for
    tuned_controller.I_COEFF_FOR = tuned_I_coeff_for
    tuned_controller.D_COEFF_FOR = tuned_D_coeff_for
    tuned_controller.P_COEFF_TOR = tuned_P_coeff_tor
    tuned_controller.I_COEFF_TOR = tuned_I_coeff_tor
    tuned_controller.D_COEFF_TOR = tuned_D_coeff_tor

    tuned_ctrl = [tuned_controller for _ in range(DEFAULT_NUM_DRONES)]

    # optimal controller
    optimal_controller = DSLPIDControl(drone_model=DEFAULT_DRONES)
    optimal_ctrl = [optimal_controller for _ in range(DEFAULT_NUM_DRONES)]

    simulate_trajectory(ctrl_env, tuned_ctrl, target_test_trajectory, name="tuned")
    simulate_trajectory(ctrl_env, optimal_ctrl, target_test_trajectory, name="optimal")


if __name__ == "__main__":
    tune_pid_coeffs()
    plot()
