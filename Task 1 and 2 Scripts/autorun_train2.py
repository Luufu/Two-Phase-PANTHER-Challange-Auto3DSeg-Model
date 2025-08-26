from monai.apps.auto3dseg import AutoRunner
from datetime import datetime

def main():
    runner = AutoRunner(
        input={
            "modality": "MRI",  # your task uses MRI
            # 👉 point these to your PRED-CROPPED dataset folder
            "dataroot": "/home/keshav/PANTHER_Task2_Auto3DSeg/data_stage2_predcrop",
            "datalist": "/home/keshav/PANTHER_Task2_Auto3DSeg/data_stage2_predcrop/dataset.json",

            # IMPORTANT: matches your label mapping (0=bg, 1=tumor, 2=pancreas)
            "class_names": ["tumor", "pancreas"],

            # train config
            "num_epochs": 650,
            "calc_val_loss": True,
            "validate_interval": 5,
            "batch_size": 1,
            "amp": True,
        },
        algos="segresnet",
        work_dir="/home/keshav/PANTHER_Task2_Auto3DSeg/results_stage2"
    )

    # (optional) keep explicit override
    train_param = {
        "num_epochs": 750,
        "validate_interval": 5,
        "batch_size": 1,
        "amp": True,
    }
    runner.set_training_params(train_param)

    print("🚀 Starting Stage-2 (pred-cropped) training")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    runner.run()

if __name__ == "__main__":
    main()
