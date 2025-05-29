import os
import time
import subprocess
import psutil

# set your target pid and bash command here
target_pid = 77531  # replace with actual pid
bash_command = "echo 'process done' && touch done.txt"  # replace with your command

def process_is_running(pid):
    return psutil.pid_exists(pid) and psutil.Process(pid).is_running()

while process_is_running(target_pid):
    time.sleep(2)  # check every 2 seconds

# process is done, run the command
subprocess.run(bash_command, shell=True)