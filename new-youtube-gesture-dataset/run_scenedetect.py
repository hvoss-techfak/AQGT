# ------------------------------------------------------------------------------
# Copyright 2019 ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from __future__ import unicode_literals
import subprocess
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from random import shuffle

from tqdm import tqdm
from config import my_config


def run_pyscenedetect(file_path, vid):  # using Pyscenedetect
    cmd = 'scenedetect --input "{}" --output "{}" -d 4 detect-content list-scenes'.format(file_path, os.path.abspath(
        my_config.CLIP_PATH))
    os.chdir(my_config.VIDEO_PATH)
    print(my_config.CLIP_PATH)
    print('  ' + cmd)
    subprocess.run(cmd, shell=True, check=True)
    subprocess.run("exit", shell=True, check=True)

def doThread(file_path):
    if file_path.endswith(my_config.FILETYPE):
            file_path = os.path.abspath(file_path)
            #print('{}/{}'.format(i + 1, n_total))
            vid = os.path.basename(file_path).replace(my_config.FILETYPE,"")
            csv_files = glob.glob(my_config.CLIP_PATH + "/{}*.csv".format(vid))
            if len(csv_files) > 0 and os.path.getsize(csv_files[0]):  # existing and not empty
                print('  CSV file already exists ({})'.format(vid))
            else:
                run_pyscenedetect(file_path, vid)

def main():
    if not os.path.exists(my_config.CLIP_PATH):
        os.makedirs(my_config.CLIP_PATH)

    print(my_config.VIDEO_PATH + "/*"+my_config.FILETYPE)

    videos = glob.glob(my_config.VIDEO_PATH + "/*"+my_config.FILETYPE)
    n_total = len(videos)
    print(n_total)
    dolist = []
    tlist = []
    exec = ProcessPoolExecutor()
    for i, file_path in tqdm(enumerate(sorted(videos, key=os.path.getmtime)),total=n_total):
        file_path1 = os.path.abspath(file_path)
        vid = os.path.basename(file_path1).replace(my_config.FILETYPE, "")
        ee = my_config.CLIP_PATH + "/{}-Scenes.csv".format(vid)

        print(ee)
        if not os.path.exists(ee):
            dolist.append((i, file_path))
        else:
            print("existing")
    shuffle(dolist)
    for i,file_path in dolist:
        tlist.append(exec.submit(doThread,file_path))
    for t in tqdm(tlist):
        t.result()


if __name__ == '__main__':
    main()
