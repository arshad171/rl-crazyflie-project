import logging
import time

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

URI = 'radio://0/80/2M'
# PID / MEL
MODE = "MEL"

def param_stab_est_callback(name, value):
    print('The crazyflie has parameter ' + name + ' set at number: ' + value)

if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI) as scf:
        if MODE == "PID":
            commander_en = 0
            stabilizer_contoller = 1
        elif MODE == "MEL":
            commander_en = 1
            stabilizer_contoller = 2

        print("-" * 5, "writing params", "-" * 5)
        print(f"{commander_en=}")
        print(f"{stabilizer_contoller=}")

        scf.cf.param.set_value('commander.enHighLevel', commander_en)
        time.sleep(1)
        scf.cf.param.set_value('stabilizer.controller', stabilizer_contoller)
        time.sleep(1)
        
        print("-" * 5, "reading params", "-" * 5)
        print(scf.cf.param.get_value('commander.enHighLevel'))
        print(scf.cf.param.get_value('stabilizer.controller'))
