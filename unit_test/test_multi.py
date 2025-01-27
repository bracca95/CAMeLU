import os
import time
import subprocess

from glob import glob

if __name__=="__main__":

    # for all the config files available
    for config_file in sorted(glob(os.path.join(os.getcwd(), "unit_test", "*"))):
        if not config_file.endswith(".json"):
            continue
        
        # run the main program, waiting for the subprocess to end
        process = subprocess.Popen(["python", "main.py", "--config_file", f"{config_file}"])
        process.wait()

        if process.returncode == 0:
            # give enough time to wandb to close
            time.sleep(10)

            # at the end of the execution:
            #   1. create a folder with the current experiment name
            experiment_out_folder = os.path.join(os.getcwd(), "output", os.path.basename(config_file).rsplit(".json", -1)[0])
            if not os.path.exists(experiment_out_folder):
                os.makedirs(experiment_out_folder)

            #   2. move all output files (.pth, .log) to the newly created directory
            outfiles = list(filter(lambda x: not os.path.isdir(x), glob(os.path.join(os.getcwd(), "output", "*"))))
            for file in outfiles:
                full_dst_file = os.path.join(experiment_out_folder, os.path.basename(file))
                os.rename(file, full_dst_file)

            #   3. copy original config file to output folder
            os.rename(config_file, os.path.join(experiment_out_folder, os.path.basename(config_file)))
            continue

        print(f"ERROR in process {os.path.basename(config_file)}")