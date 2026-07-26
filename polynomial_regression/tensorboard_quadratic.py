import numpy as np
np.object = np.object_
np.bool = np.bool_
np.int = np.int_
from tensorboard import program
import os

# os.chdir("")
tracking_address = 'logs/fit/' # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

# Takes a while to load everything
