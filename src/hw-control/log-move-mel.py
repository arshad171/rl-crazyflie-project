"""
This script shows a simple scripted flight path using the MotionCommander class.

Simple example that connects to the crazyflie at `URI` and runs a
sequence. Change the URI variable to your Crazyflie configuration.
"""
import logging
import time
import numpy as np

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie, Crazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander

from cflib.utils import uri_helper

#URI = "radio://0/80/2M"
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

def simple_log(scf, lg_stab):
    with SyncLogger(scf, lg_stab) as logger:
        for log_entry in logger:
            timestamp = log_entry[0]
            data = log_entry[1]
            logconf_name = log_entry[2]

            print("%s" % (data))

            break

def wait_for_position_estimator(scf):
    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)



if __name__ == "__main__":
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    lg_stab = LogConfig(name='Stabilizer', period_in_ms=100)
    lg_stab.add_variable('stateEstimate.x', 'float')
    lg_stab.add_variable('stateEstimate.y', 'float')
    lg_stab.add_variable('stateEstimate.z', 'float')
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')
    lg_stab.add_variable('pm.vbat', 'FP16')

    initial_height = 0.2

    num_samples = 30
    helix_rad = 0.3
    helix_height = 0.5
    angles = np.linspace(0, 4 * np.pi, num_samples)
    trajectory = np.zeros((num_samples, 3))  # x, y, z

    trajectory[:, 0] = helix_rad * np.cos(angles)
    trajectory[:, 1] = helix_rad * np.sin(angles)
    trajectory[:, 2] = np.linspace(
        initial_height, initial_height + helix_height, num_samples
    )

    duration = np.zeros(shape=(num_samples))
    target_velocity = 0.06

    init_pos = np.array([[0.0, 0.0, initial_height]])

    trajectory_diff = np.diff(trajectory, axis=0, prepend=init_pos)

    distance = np.linalg.norm(trajectory_diff, axis=1)

    for i in range(num_samples):
        duration[i] = distance[i] / target_velocity

    # print(trajectory_diff)

    # print(duration)

    # print(np.sum(duration))

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        reset_estimator(cf)
        time.sleep(3)

        commander = scf.cf.high_level_commander

        commander.takeoff(initial_height, 1)
        time.sleep(5)
        simple_log(scf, lg_stab)

        for (pos, d) in zip(trajectory_diff, duration):
            commander.go_to(pos[0], pos[1], pos[2], 0, duration_s=d, relative=True)
            time.sleep(d)
            simple_log(scf, lg_stab)

        commander.land(0.0, 1)
        time.sleep(5)

        commander.stop()
