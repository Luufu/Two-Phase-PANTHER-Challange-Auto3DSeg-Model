# from monai.apps.auto3dseg import AutoRunner
# from datetime import datetime
# import os

# def main():
#     runner = AutoRunner(
#         input={
#             "modality": "CT",  # ✅ PANTHER Task 1 uses CT
#             "dataroot": "/home/keshav/PANTHER_Task1_Auto3DSeg/data",
#             "datalist": "/home/keshav/PANTHER_Task1_Auto3DSeg/data/dataset.json",
#             "class_names": ["pancreas", "tumor"],
#             "num_epochs": 50,  # or set this below in train_param
#         },
#         algos="segresnet",  # ✅ This is the backbone
#         work_dir="/home/keshav/PANTHER_Task1_Auto3DSeg/results"
#     )

#     # Optional: override some training settings
#     train_param = {
#         "num_epochs": 50,
#     }
#     runner.set_training_params(train_param)

#     print("🚀 Starting AutoRunner training")
#     print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#     runner.run()

# if __name__ == "__main__":
#     main()


from monai.apps.auto3dseg import AutoRunner
from datetime import datetime
import os

def main():
    runner = AutoRunner(
        input={
            "modality": "MRI",  #  PANTHER Task 2 uses MRI
            "dataroot": "/home/keshav/PANTHER_Task2_Auto3DSeg/data_cropped",
            "datalist": "/home/keshav/PANTHER_Task2_Auto3DSeg/data_cropped/dataset.json",
            "class_names": ["tumor", "pancreas"],
            "num_epochs": 1000,
            "calc_val_loss": True,
            "validate_interval": 5,
            "batch_size": 1,
            "amp": True,
        },
        algos="segresnet",
        work_dir="/home/keshav/PANTHER_Task2_Auto3DSeg/results"
    )

    # ✅ Correct training hyperparameters
    train_param = {
        "num_epochs": 1000,
        "validate_interval": 5,
        "batch_size": 1,
        "amp": True,
    }

    runner.set_training_params(train_param)

    print("🚀 Starting AutoRunner training")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    runner.run()

if __name__ == "__main__":
    main()


