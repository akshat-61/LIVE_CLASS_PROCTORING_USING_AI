import shutil
from config import *

def deploy(new_model):

    shutil.copy(
        new_model,
        f"{MODEL_PRODUCTION}/object_model.pt"
    )

    print("🚀 New model deployed")