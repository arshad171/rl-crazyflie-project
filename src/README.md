# RL Control of CF

## Project Structure

- All the source files go into `src` dir.

- Any interesting results/observations can be copied to [results](../results/) folder.

## PID Tuning

1. Nagivate to [pid-tuning](./rl/pid-tuning/).

2. Run `python3 pid-tuning.py`. This will train the model and generate results in the `results` dir comparing the performance of tuned and optmial PID controller. The file `Logger` is a modified version of `gym-pybullet-drones/gym_pybullet_drones/utils/Logger.py` to enable logging of reference signals for comparison purposes. `plotter` plots the responses (XYZ, RPY) of randomly initialized PID coeffs, tuned and optimal (default on hardware) PID coeffs.

3. The results are copied to [results folder](../results/rl/pid-tuning/).

## Objectives/Ideas (Brainstorming)

1. [simulation] PID tuning using RL (`stable_baselines3`: A2C, DDPG, TD3).
    1. Modeling: CF 2.X/P URDF
1. Testing the coeffs on real hardware.
    1. Figure out compatibility.
    1. The task could be trajectory following such a circle or helix.
1. Have a custom DDPG/TD3 implementations.
1. Navigation: PID for maneuvers and RL for navigating to the target location.
    1. [simulation] Create and environment, where the task would be to navigate to the specified location.
        - Observation space: limited to position (XYZ) and orientation (RPY).
        - Action space:
            - To begin with, the action space may be discrete and comprise of simple motion primitives such as move forward, left, right, up, down, forward, backward, etc to evaluate convergence and feasibility of the task.
            - The next step would be to have continuous action space where the actions correspond to movements in a cube (relative).
        - The learnings from the previous task (Mellinger PID) could be used execute the actions. So the policy would be responsible to output the respective actions.
        - Compute rewards based on actions performed and target location.
    1. Integrate lighthouse positioning.
    1. Running on real hardware. AI deck for running on board NN computations or over radio.
    1. Navigation approach using discrete motion planning and constant velocity motion primitives:
        -  A state lattice space is drawn with a predefined grid, designed using Crazyflie's relative positioning (flow) or absolute positioning (lighthouse)
        -  Obstacles are mapped to the lattice, where an array of actions from start to goal is obtained using Q-Learning, SARSA, Dijkstras or A* (shortest path problem)
        - The solution algorithm is computed for different sets of motion primitives (e.g., different velocity or available actions in the 3D space)

## Tasks 1-3: PID Tuning

- **Observation Space**:
    - length = 12
    - range: [-1, 1]
    - kinematics based observation or state representation. position (3), orientation (3), linear and angular velocities (3 + 3).

- **Action Space**: 
    - length = 6
    - range: [-1, 1]
    - represent the multipliers for the PID coeffs. `TUNED_P_POS = (action[0] + 1) * TUNED_P_POS`. The same multiplier affects all the three coeffs (XYZ) per control. $+1$ due to tanh activation, converts the range to $[0, 2]$.

The idea is to optimize the PID coeffs so that the computed control (based on the dymanics of CF2) results in a better maneuver to the desired position in the trajectory. $(s, a) \to (s', r)$ and the reward is $(s' - t)^2$, where $t$ is the target position in the trajectory.

## Task 4: Navigation

- State: kinematics observation.
- Cost: +1 for moving away from the destination, -1 for moving closer to the destination. Can also be a heuristic-based cost.
- Actor responsible for outputing the action (forward, left, right, up, down, forward, backward). Actor cost: $Q(s, \mu(s))$
- Critic cost: $(Q(s, a) - (g + \gamma * (1 - \delta) Q_t(s', \mu_t(a'))))^2$


### Coefficients

#### Unstable (rand init, untrained, should not be tested on real hardware)

```
tune_env.ctrl.P_COEFF_FOR=array([0.76338086, 0.76338086, 2.38556519])
tune_env.ctrl.I_COEFF_FOR=array([0.07520749, 0.07520749, 0.07520749])
tune_env.ctrl.D_COEFF_FOR=array([0.15180122, 0.15180122, 0.37950304])
tune_env.ctrl.P_COEFF_TOR=array([71647.83295244, 71647.83295244, 61412.42824495])
tune_env.ctrl.I_COEFF_TOR=array([  0.        ,   0.        , 895.56157589])
tune_env.ctrl.D_COEFF_TOR=array([32016.17479324, 32016.17479324, 19209.70487595])
```

#### Stable (tuned, can be tested on read hardware)

```
tune_env.ctrl.P_COEFF_FOR=array([0.37399905, 0.37399905, 1.16874702])
tune_env.ctrl.I_COEFF_FOR=array([0.05232859, 0.05232859, 0.05232859])
tune_env.ctrl.D_COEFF_FOR=array([0.23473141, 0.23473141, 0.58682853])
tune_env.ctrl.P_COEFF_TOR=array([64786.8424654 , 64786.8424654 , 55531.57925606])
tune_env.ctrl.I_COEFF_TOR=array([  0.        ,   0.        , 599.66611862])
tune_env.ctrl.D_COEFF_TOR=array([17217.40603447, 17217.40603447, 10330.44362068])
```

#### Default (optimal)

```
self.P_COEFF_FOR = np.array([.4, .4, 1.25])
self.I_COEFF_FOR = np.array([.05, .05, .05])
self.D_COEFF_FOR = np.array([.2, .2, .5])
self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
self.I_COEFF_TOR = np.array([.0, .0, 500.])
self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
```

#### Mellinger Controller - HW Config

Enable high-level commander and set Mellinger controller.

```
commander.enHighLevel: 0 -> 1
stabilizer.controller: 1 -> 2
```

Controller params:

The params can be found in the param group `ctrlMel`

```
# position control
kp_xy, kp_z
ki_xy, ki_z
kd_xy, kd_z

# att control
kR_xy, kR_z
ki_m_xy, ki_m_z
kw_xy, kw_z
```

### Navigation: Executing Control Loop to Execute Policy Action 

1. Replace the following line (326) in `gym_pybullet_drones/envs/BaseAviary.py` with the following block of code.
1. Replace the `_physics` function definition in `gym_pybullet_drones/envs/BaseAviary.py`.

**IMPORTANT, Set** `AGGR_PHY_STEPS = 1, NUM_PHYSICS_STEPS = 1`.

```python
# calc dist
if self.flip_freq == -1:
    dist = self.ext_dist_mag
else:
    dist = self.ext_dist_mag[self.ext_dist_index, :]
    if self.step_counter != 0 and self.step_counter % self.flip_freq == 0:
        self.ext_dist_index = (self.ext_dist_index + 1) % self.ext_dist_mag.shape[0]

# execute pid loop
state = self._getDroneStateVector(0)
target = np.array(state[0:3]+0.05*action)
last_control = None
AGGR_PHY_STEPS = 1
CONTROL_DURATION_SEC = 100
CONTROL_FREQ = 48
CONTROL_EVERY_N_STEPS = int(np.floor(self.SIM_FREQ / CONTROL_FREQ))
for c in range(0, int(CONTROL_DURATION_SEC * self.SIM_FREQ / CONTROL_FREQ), AGGR_PHY_STEPS):
    if c % CONTROL_EVERY_N_STEPS == 0:
        # compute the control
        curr_state = self._getDroneStateVector(0)
        rpm, _, _ = self.ctrl.computeControl(
            control_timestep=CONTROL_EVERY_N_STEPS*self.TIMESTEP, 
            cur_pos=curr_state[0:3],
            cur_quat=curr_state[3:7],
            cur_vel=curr_state[10:13],
            cur_ang_vel=curr_state[13:16],
            target_pos=target,
        )
        last_control = np.reshape(rpm, (self.NUM_DRONES, 4))
    self._physics(last_control[i, :], i, dist=dist)
    p.stepSimulation(physicsClientId=self.CLIENT)
    self._updateAndStoreKinematicInformation()
```

```python
def _physics(self,
                 rpm,
                 nth_drone,
                 dist=None
                 ):
    """Base PyBullet physics implementation.

    Parameters
    ----------
    rpm : ndarray
        (4)-shaped array of ints containing the RPMs values of the 4 motors.
    nth_drone : int
        The ordinal number/position of the desired drone in list self.DRONE_IDS.

    """
    forces = np.array(rpm**2)*self.KF
    torques = np.array(rpm**2)*self.KM
    z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
    for i in range(4):
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                i,
                                forceObj=[0, 0, forces[i]],
                                posObj=[0, 0, 0],
                                flags=p.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
    p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                            4,
                            torqueObj=[0, 0, z_torque],
                            flags=p.LINK_FRAME,
                            physicsClientId=self.CLIENT
                            )
    if dist is not None:
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                -1,
                                forceObj=dist,
                                posObj=[0, 0, 0],
                                flags=p.LINK_FRAME,
                                physicsClientId=self.CLIENT
                                )
        # p.applyExternalTorque(self.DRONE_IDS[nth_drone],
        #                       -1,
        #                       torqueObj=[0.0, 0.0, 0.0001],
        #                       flags=p.LINK_FRAME,
        #                       physicsClientId=self.CLIENT
        #                       )
```

### Adding External Disturbance

Append the following to `_physics` function in `gym_pybullet_drones/envs/BaseAviary.py`

```python
p.applyExternalForce(self.DRONE_IDS[nth_drone],
                        -1,
                        forceObj=[0.05, 0.05, 0.05],
                        posObj=[0, 0, 0],
                        flags=p.LINK_FRAME,
                        physicsClientId=self.CLIENT
                        )
# p.applyExternalTorque(self.DRONE_IDS[nth_drone],
#                       -1,
#                       torqueObj=[0.0, 0.0, 0.0001],
#                       flags=p.LINK_FRAME,
#                       physicsClientId=self.CLIENT
#                       )
```

### Fix Recording

Currently the `BaseAviary._startVideoRecording` fails to record the video complaing about a missing temporary directy that it intends to create. The following code fixes the issue, and also runs a counter to save the video for every 10 epochs.

**If the library is installed in editable mode, simply replace the source with the following snippet**

```python
def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        self.rec_counter = self.rec_counter + 1

        if not self.rec_counter % 10 == 0:
            return

        self.rec_counter = 0

        if self.RECORD and self.GUI:
            path = os.path.join(self.OUTPUT_FOLDER, "rl-train", "rec", "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"), "output.mp4")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=path,
                                                physicsClientId=self.CLIENT
                                                )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"), '')
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
```