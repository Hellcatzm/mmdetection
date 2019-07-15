import os
import subprocess

print(os.listdir('./'))
assert os.path.isfile('./tools/dist_train.sh'), "Can't found file 'dist_train.sh'"

commands = [
    "./tools/dist_train.sh configs/carbonate/htc_libra_cca_dconv2_c3-c5_se_x101_64x4d_pan.py 2",
    "./tools/dist_train.sh configs/carbonate/trident/htc_trident.py 2",
    "./tools/dist_train.sh configs/carbonate/trident/trident_c4c5_r50_fpn_1x.py 2",
]

for cmd in commands:
    print("Process command '{}' ...".format(cmd))
    try:
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
    except RuntimeError as e:
        continue