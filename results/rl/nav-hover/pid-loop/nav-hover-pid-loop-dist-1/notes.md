NUM_EPISODES = 5e5
ACTOR_NET_ARCH = [50, 100, 500, 100, 50]
CRITIC_NET_ARCH = [50, 100, 500, 100, 50]
sigma = 0.5
action scale = 0.05
DEFAULT_SIMULATION_FREQ_HZ = 50
DEFAULT_DURATION_SEC = 2

TRAIN_EXT_DIST = np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.01], [0.01, 0.01, 0.01]])
FLIP_FREQ = -1 if MODE == "test" else 5

NavigationAviary:
self.EPISODE_LEN_SEC = 2