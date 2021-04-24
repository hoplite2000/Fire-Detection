import os

Fire_path = ""
Non_fire_path = ""

Classes = ["Non-Fire", "Fire"]

Train_split = 0.75
Test_split = 0.25

Init_lr = 1e-2
Batch_size = 64
Epochs = 50

Model_path = os.path.sep.join(["output", "fire_detection.model"])
Lr_find_plot_path = os.path.sep.join(["output", "lrfind_plot.png"])
Training_plot_path = os.path.sep.join(["output", "training_plot.png"])

OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])
SAMPLE_SIZE = 50