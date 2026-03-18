import os
import shutil
from config import *

def build_dataset():

    os.makedirs(TRAIN_IMAGES, exist_ok=True)
    os.makedirs(TRAIN_LABELS, exist_ok=True)

    for category in os.listdir(CAPTURED_FRAMES):

        folder = os.path.join(CAPTURED_FRAMES, category)

        for file in os.listdir(folder):

            if file.endswith(".jpg"):

                src = os.path.join(folder, file)
                dst = os.path.join(TRAIN_IMAGES, file)

                shutil.copy(src, dst)

    print("✅ Dataset built")