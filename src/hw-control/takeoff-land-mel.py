"""
This script shows a simple scripted flight path using the MotionCommander class.

Simple example that connects to the crazyflie at `URI` and runs a
sequence. Change the URI variable to your Crazyflie configuration.
"""
import contextlib
import logging
import time

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie, Crazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander

URI = 'radio://0/80/2M'

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

def log_stab_callback(timestamp, data, logconf):
    # print('[%d][%s]: %s' % (timestamp, logconf.name, data))
    print('%s' % (data))

def simple_log_async(scf, logconf):
    cf = scf.cf
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(log_stab_callback)
    # logconf.start()
    # time.sleep(5)
    # logconf.stop()


if __name__ == '__main__':
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

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        commander = scf.cf.high_level_commander

        simple_log_async(scf, lg_stab)
        lg_stab.start()

        commander.takeoff(0.2, 1)
        time.sleep(3)
        commander.land(0.2, 1)
        time.sleep(3)

        lg_stab.stop()
        commander.stop()
