import os
import subprocess

print(os.listdir('./'))
assert os.path.isfile('./tools/dist_train.sh'), "Can't found file 'dist_train.sh'"

commands = [
    "./tools/dist_train.sh ./configs/carbonate/htc_libra_dconv2_c3-c5_se_x101_64x4d_pan.py 2",
    "./tools/dist_train.sh ./configs/carbonate/htc_libra_dconv2_c3-c5_se_x101_64x4d_pan_ms.py 2",
    "./tools/dist_train.sh configs/carbonate/htc_trident.py 2",
    "python ./tools/train.py ./configs/carbonate/htc_libra_dconv2_c3-c5_se_x101_64x4d_pan_ga.py",
]

for cmd in commands:
    print("Process command '{}' ...".format(cmd))
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
