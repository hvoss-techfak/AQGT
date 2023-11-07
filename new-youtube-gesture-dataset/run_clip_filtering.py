# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from __future__ import unicode_literals
import csv
import os.path
from concurrent.futures import ProcessPoolExecutor
from random import shuffle
from time import sleep

import numpy as np
from tqdm import tqdm

from clip_filter import *
from main_speaker_selector import *
from config import my_config

RESUME_VID = ''  # resume the process from this video


def read_sceneinfo(filepath):  # reading csv file
    with open(filepath, 'r') as csv_file:
        frame_list = [0]
        for i,row in enumerate(csv.reader(csv_file)):
            if row and i > 1 and len(row) > 0 and row[0] != "Scene Number": # header is not always two lines? sometimes three, sometimes it is weird...
                frame_list.append((row[1]))

    frame_list = [int(x) for x in frame_list]  # str to int

    return frame_list


def run_filtering(scene_data, skeleton_wrapper, video_wrapper):
    filtered_clip_data = []
    aux_info = []
    video = video_wrapper.get_video_reader()
    #vis = VisualizeCvClass()


    for i in range(len(scene_data)):
        start_frame_no, end_frame_no = scene_data[i], (scene_data[i + 1] if i+1 < len(scene_data) else skeleton_wrapper.getLen())
        if end_frame_no-start_frame_no > 1:
            raw_skeleton_chunk = skeleton_wrapper.get(start_frame_no, end_frame_no)

            main_speaker_skeletons = MainSpeakerSelector(raw_skeleton_chunk=raw_skeleton_chunk).get()

            # run clip filtering
            clip_filter = ClipFilter(video=video, start_frame_no=start_frame_no, end_frame_no=end_frame_no,
                                     raw_skeleton=raw_skeleton_chunk, main_speaker_skeletons=main_speaker_skeletons)
            correct_clip = clip_filter.is_correct_clip()

            filtering_results, message, debugging_info = clip_filter.get_filter_variable()
            filter_elem = {'clip_info': [start_frame_no, end_frame_no, correct_clip], 'filtering_results': filtering_results,
                           'message': message, 'debugging_info': debugging_info}
            aux_info.append(filter_elem)

            #if correct_clip:
                #for i in range(len(main_speaker_skeletons)):
                #    if len(list(main_speaker_skeletons[i].keys())) > 0:
                #        skeleton = main_speaker_skeletons[i]["current_3d_pose"]
                #        cv2.imshow("img", vis.VisUpperBody(skeleton, np.ones(53, )))
                #        cv2.waitKey(33)
            #    print("yes")
            #else:
            #    print("no")

            # save
            elem = {'clip_info': [start_frame_no, end_frame_no, correct_clip], 'frames': []}

            if not correct_clip:
                filtered_clip_data.append(elem)
                continue
            elem['frames'] = convertToLists(main_speaker_skeletons)
            filtered_clip_data.append(elem)

    return filtered_clip_data, aux_info

def convertToLists(main_speaker_skeletons):
    rets = []
    for main_s in main_speaker_skeletons:
        ret = {}
        if "id" in main_s:
            ret["id"] = int(main_s["id"])
            ret["current_2d_score"] = main_s["current_2d_score"].tolist()
            ret["current_3d_pose"] = main_s["current_3d_pose"].tolist()
        rets.append(ret)
    return rets
    #for k in main_speaker_skeletons.keys():

def doThread(csv_path):
    vid = os.path.basename(csv_path)[:-11] + my_config.FILETYPE
    vid1 = os.path.basename(csv_path)[:-11]

    skip_flag = False
    out_clips = 0
    #print("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid))
    try:
        if not os.path.exists("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid1)):

            if os.path.exists(my_config.VIDEO_PATH + '/' + vid and os.path.exists(my_config.VIDEO_PATH + '/' + vid + '_calc.kp')):

                #print(csv_path)
                #print(vid)
                #tqdm.write(vid)
                try:
                    if not skip_flag:
                        scene_data = read_sceneinfo(csv_path)
                        skeleton_wrapper = AlphaSkeletonWrapper(my_config.SKELETON_PATH, vid)
                        video_wrapper = read_video(my_config.VIDEO_PATH, vid)

                        if video_wrapper.height < 480:
                            print('[Fatal error] wrong video size (height: {})'.format(video_wrapper.height))
                            print(vid)
                            return 0

                        if len(skeleton_wrapper.skeletons) == 0:
                            return 0
                        if abs(video_wrapper.total_frames - len(skeleton_wrapper.skeletons)) > 10:
                            print('[Fatal error] video and skeleton object have different lengths (video: {}, skeletons: {})'.format
                                  (video_wrapper.total_frames, len(skeleton_wrapper.skeletons)))
                            print(vid)
                            return 0

                        if skeleton_wrapper.skeletons == [] or video_wrapper is None:
                            print('[warning] no skeleton or video! skipped this video.')
                            print(vid)
                        else:
                            ###############################################################################################
                            filtered_clip_data, aux_info = run_filtering(scene_data, skeleton_wrapper, video_wrapper)
                            ###############################################################################################
                            for c in filtered_clip_data:
                                if len(c['frames']) > 0:
                                    out_clips += 1
                            # save filtered clips and aux info
                            with open("{}/{}.json".format(my_config.CLIP_PATH, vid1), 'w') as clip_file:
                                json.dump(filtered_clip_data, clip_file)
                            with open("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid1), 'w') as aux_file:
                                json.dump(aux_info, aux_file)
                            return out_clips
                except Exception:
                    import traceback
                    traceback.print_exc()
                    return 0
        else:
            print("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid), "aux info already exists")
    except Exception:
        import traceback
        traceback.print_exc()
    return 0

def main():
    if RESUME_VID == "":
        skip_flag = False
    else:
        skip_flag = True

    tlist = []
    cout = 0
    exec = ProcessPoolExecutor(max_workers=8)
    l1 = sorted(glob.glob(my_config.CLIP_PATH + "/*.csv"), key=os.path.getmtime)
    shuffle(l1)
    for csv_path in tqdm(l1):
        vid = os.path.basename(csv_path)[:-11]
        if not os.path.exists("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid)):
            tlist.append(exec.submit(doThread,csv_path))
            #doThread(csv_path)
    # while len(tlist) > 0:
    #     for t in tlist.copy():
    #         if t.done():
    #             tlist.remove(t)
    #     sleep(5)
    #     print("remaining:",len(tlist))
    tbar = tqdm(total=len(tlist))
    while len(tlist) > 0:
        for t in tlist:
            if t.done():
                cc = t.result()
                if cc is not None:
                    cout += cc
                tbar.set_description(str(cout))
                tbar.update(1)
                tlist.remove(t)
        sleep(1)
    exec.shutdown()

if __name__ == '__main__':
    main()
