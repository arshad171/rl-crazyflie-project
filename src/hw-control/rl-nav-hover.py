import sys
sys.path.append('./src')

import logging
import pickle
import time
import numpy as np

from stable_baselines3 import TD3, A2C

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie, Crazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander

from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

MODEL_PATH = "./results/rl-train/model"
ENV_PATH = "./results/rl-train/env"

DELAY = 2

logging.basicConfig(level=logging.ERROR)


def simple_log(scf, lg):
    with SyncLogger(scf, lg) as logger:
        for log_entry in logger:
            timestamp = log_entry[0]
            data = log_entry[1]
            logconf_name = log_entry[2]

            print("%s" % (data))

            return data


def wait_for_position_estimator(scf):
    log_config = LogConfig(name="Kalman Variance", period_in_ms=500)
    log_config.add_variable("kalman.varPX", "float")
    log_config.add_variable("kalman.varPY", "float")
    log_config.add_variable("kalman.varPZ", "float")

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data["kalman.varPX"])
            var_x_history.pop(0)
            var_y_history.append(data["kalman.varPY"])
            var_y_history.pop(0)
            var_z_history.append(data["kalman.varPZ"])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (
                (max_x - min_x) < threshold
                and (max_y - min_y) < threshold
                and (max_z - min_z) < threshold
            ):
                break


def reset_estimator(cf):
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")

    wait_for_position_estimator(cf)


if __name__ == "__main__":
    nav_env = pickle.load(open(ENV_PATH, "rb"))
    model = TD3.load(MODEL_PATH, nav_env)

    cflib.crtp.init_drivers(enable_debug_driver=False)

    lg_stab = LogConfig(name="Stabilizer", period_in_ms=50)
    lg_state = LogConfig(name="stateEstimate", period_in_ms=50)
    lg_gyro = LogConfig(name="gyro", period_in_ms=50)

    lg_state.add_variable("stateEstimate.x", "float")
    lg_state.add_variable("stateEstimate.y", "float")
    lg_state.add_variable("stateEstimate.z", "float")
    lg_state.add_variable("stateEstimate.vx", "float")
    lg_state.add_variable("stateEstimate.vy", "float")
    lg_state.add_variable("stateEstimate.vz", "float")

    lg_stab.add_variable("stabilizer.roll", "float")
    lg_stab.add_variable("stabilizer.pitch", "float")
    lg_stab.add_variable("stabilizer.yaw", "float")

    lg_gyro.add_variable("gyro.x", "float")
    lg_gyro.add_variable("gyro.y", "float")
    lg_gyro.add_variable("gyro.z", "float")


    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        reset_estimator(cf)
        time.sleep(3)

        commander = scf.cf.high_level_commander

        # begin loop
        while True:
            # measure state
            measurements_state = simple_log(scf, lg_state)
            time.sleep(0.1)
            measurements_stab = simple_log(scf, lg_stab)
            time.sleep(0.1)
            measurements_gyro = simple_log(scf, lg_gyro)
            time.sleep(0.1)
            # print(measurements_state, measurements_stab, measurements_gyro)
            # print("*****")
            state = np.array(
                [
                    [
                        measurements_state["stateEstimate.x"],
                        measurements_state["stateEstimate.y"],
                        measurements_state["stateEstimate.z"],
                        measurements_stab["stabilizer.roll"] * np.pi / 180,
                        measurements_stab["stabilizer.pitch"] * np.pi / 180,
                        measurements_stab["stabilizer.yaw"] * np.pi / 180,
                        measurements_state["stateEstimate.vx"],
                        measurements_state["stateEstimate.vy"],
                        measurements_state["stateEstimate.vz"],
                        measurements_gyro["gyro.x"] * np.pi / 180,
                        measurements_gyro["gyro.y"] * np.pi / 180,
                        measurements_gyro["gyro.z"] * np.pi / 180,
                    ]
                ]
            )
            # model.predict
            action, _ = model.predict(state)
            action = action[0]
            action = 0.1 * action
            # print(action)
            # print("--------")
            # send action to cf (duration: delay)
            commander.go_to(
                action[0], action[1], action[2], 0, duration_s=DELAY, relative=True
            )
            # wait for delay
            time.sleep(DELAY)
