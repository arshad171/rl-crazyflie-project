from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf

class TBCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self._writer = tf.summary.create_file_writer(log_dir)
    
        self._counter = 0
    
    def _on_step(self) -> bool:
        x, y = ts2xy(load_results(self.log_dir), "timesteps")

        if y.size:
            with self._writer.as_default():
                tf.summary.scalar("eps_rewards", data=y[-1], step=self._counter)
                self._counter = self._counter + 1

            self._writer.flush()
        return True
