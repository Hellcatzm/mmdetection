import os
import subprocess

print(os.listdir('./'))
assert os.path.isfile('./tools/dist_train.sh'), "Can't found file 'dist_train.sh'"

commands = [
    # "./tools/dist_train.sh configs/carbonate/trident/faster_rcnn_r50_fpn_1x_.py 2",
    "./tools/dist_train.sh configs/carbonate/trident/trident_c4c5_r50_fpn_1x.py 2",
    # "./tools/dist_train.sh configs/carbonate/trident/trident_c4_r50_fpn_1x.py 2",
    # "./tools/dist_train.sh configs/carbonate/trident/faster_rcnn_r50_fpn_1x.py 2",
    # "./tools/dist_train.sh configs/carbonate/trident/faster_rcnn_r50_fpn_1x_single.py 2",
]

for cmd in commands:
    print("Process command '{}' ...".format(cmd))
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
