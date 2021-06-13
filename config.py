import os

Fire_path = "dataset/fire"
Non_fire_path = "dataset/non_fire"

Classes = ["Non-Fire", "Fire"]

Train_split = 0.75
Test_split = 0.25

Init_lr = 1e-2
Batch_size = 64
Epochs = 50

Model_path = os.path.sep.join(["models", "rms", "fire_detection64.model"])
Lr_find_plot_path = os.path.sep.join(["models", "lrfind_plot.png"])
Training_plot_path = os.path.sep.join(["models", "training_plot.png"])

OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])
SAMPLE_SIZE = 50