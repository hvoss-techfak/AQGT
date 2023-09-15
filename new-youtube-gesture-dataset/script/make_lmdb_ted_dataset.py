# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------


import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import lmdb
import numpy as np
import pyarrow

from moviepy.audio.io.AudioFileClip import AudioFileClip
from scipy import signal
from tqdm import tqdm_gui, tqdm
import unicodedata

from saga_dataloader import SaGA_dataloader

from data_utils import *


def read_subtitle(vid):
    postfix_in_filename = '-en.vtt'
    file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)
    if len(file_list) > 1:
        print('more than one subtitle. check this.', file_list)
        assert False
    if len(file_list) == 1:
        return WebVTT().read(file_list[0])
    else:
        return []


# turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def normalize_subtitle(vtt_subtitle):
    for i, sub in enumerate(vtt_subtitle):
        vtt_subtitle[i].text = normalize_string(vtt_subtitle[i].text)
    return vtt_subtitle

def getAudioData(vid):
    audio = AudioFileClip(vid, fps=16000)
    arr = audio.to_soundarray()
    if len(arr.shape) > 1:
        arr = arr.mean(axis=1)
    arr -= arr.min()
    arr /= arr.max() / 2
    arr -= 1
    return arr

def getSagaAnnotations(saga_loader,start_frame,end_frame):
    ret = []
    if saga_loader is not None:
        for i in range(start_frame,min(end_frame,saga_loader.__len__())):
            if saga_loader.current_frame > i:
                saga_loader.init_data()
            data = None
            while i != saga_loader.current_frame:
                data = saga_loader.__next__()

                del data['gesture_data'] # we don't need duplicate gesture data!
                if "Audio" in data.keys():
                    del data["Audio"] # same with audio data. We already have the information stored
            #if data["R_S_Semantic_Entity"] is not None:
            #    print(data["R_S_Semantic_Entity"])
            #if data["R_S_Semantic"] is not None:
            #    print(data["R_S_Semantic"])
            ret.append(data)
        return ret
    else:
        return None



def run(v_i,video_file,dataset_file):
    out_frames = []
    valid_clip_count = 0
    try:
        map_size = 1024 * 500  # in MB
        map_size <<= 20  # in B

        dataset = lmdb.open(dataset_file, map_size=map_size,lock=False)
        vid = os.path.basename(video_file)
        vid1 = os.path.basename(video_file)[:-len(my_config.FILETYPE)]
        #print(my_config.CLIP_PATH + '/' + vid1 + '_aux_info.json')
        if os.path.exists(my_config.VIDEO_PATH + '/' + vid) and \
                os.path.exists(my_config.VIDEO_PATH + '/' + vid + '_calc.kp') and \
                os.path.exists(my_config.CLIP_PATH + '/' + vid1 + '_aux_info.json'):

            # load clip, video, and subtitle
            clip_data = load_clip_data(vid1)
            if clip_data is None:
                print('[ERROR] clip data file does not exist!')
                return

            video_wrapper = read_video(my_config.VIDEO_PATH, vid)

            raw_audio = getAudioData(my_config.VIDEO_PATH + "/" + vid).flatten()
            audio_out = np.zeros((int(16000 * ((video_wrapper.total_frames // video_wrapper.framerate) + 1))))
            audio_out[:raw_audio.shape[0]] = raw_audio

            print(my_config.VIDEO_PATH + '/' + vid)
            try:
                saga_loader = SaGA_dataloader(my_config.VIDEO_PATH + '/' + vid)
            except Exception:
                print('[WARNING] No SaGA data found. Not adding annotation information')
                saga_loader = None
                pass

            subtitle_type = my_config.SUBTITLE_TYPE
            subtitle = SubtitleWrapper(vid1, subtitle_type,saga_loader.english_sub_file if saga_loader is not None else None).get()

            if subtitle is None:
                print('[WARNING] subtitle does not exist! skipping this video.')
                return




            #os.makedirs(dataset, exist_ok=True)

            clips = []

            word_index = 0
            for ia, clip in enumerate(clip_data):
                try:
                    start_frame_no, end_frame_no, clip_pose_all = clip['clip_info'][0], clip['clip_info'][1], clip['frames']
                    clip_word_list = []

                    a_start = int(video_wrapper.frame2second(start_frame_no) * 16000)
                    a_end = int(video_wrapper.frame2second(end_frame_no) * 16000)

                    audio_raw = audio_out[a_start:a_end]

                    # skip FALSE clips
                    if not clip['clip_info'][2]:
                        continue

                    # train/val/test split

                    # get subtitle that fits clip
                    for ib in range(word_index - 1, len(subtitle)):
                        if ib < 0:
                            continue

                        word_s = video_wrapper.second2frame(subtitle[ib]['start'])
                        word_e = video_wrapper.second2frame(subtitle[ib]['end'])
                        word = subtitle[ib]['word']

                        if word_s >= end_frame_no:
                            word_index = ib
                            break

                        if word_e <= start_frame_no:
                            continue

                        word = normalize_string(word)
                        clip_word_list.append([word, word_s, word_e])

                    if clip_word_list:
                        clip_skeleton = []

                        # get skeletons of the upper body in the clip
                        for frame in clip_pose_all:
                            if frame:
                                clip_skeleton.append(np.asarray(frame["current_3d_pose"]))
                            else:  # frame with no skeleton. We insert the last element if we have one
                                if len(clip_skeleton) > 0:
                                    clip_skeleton.append(clip_skeleton[-1].copy())
                                else:
                                    clip_skeleton.append(np.zeros((53,3)))

                        # proceed if skeleton list is not empty
                        if len(clip_skeleton) > 0:
                            # save subtitles and skeletons corresponding to clips
                            valid_clip_count += 1
                            #vis = VisualizeCvClass()
                            #for i in range(len(clip_skeleton)):
                            #    cv2.imshow("img",vis.VisUpperBody(clip_skeleton[i],np.ones(53,)))
                            #    print(clip_skeleton[i].min(),clip_skeleton[i].max())
                            #    cv2.waitKey(33)
                            try:
                                #there is ONE sample that refuses to align with the other data. quick fix for this:
                                exception_check = np.asarray(clip_skeleton).reshape((-1,53,3))
                                clip['skeletons_3d'] = clip_skeleton  # clip_data (-1, 48,3)
                                clip['audio_feat'] = np.zeros((1,))  # spectogram?
                                clip['audio_raw'] = audio_raw.tolist()  # clip_audio at 16k hz (

                                clip['words'] = clip_word_list
                                clip['start_frame_no'] = start_frame_no
                                clip['end_frame_no'] = end_frame_no

                                clip['saga_annotation'] = getSagaAnnotations(saga_loader,start_frame_no,end_frame_no)

                                clip['start_time'] = video_wrapper.frame2second(start_frame_no)
                                clip['end_time'] = video_wrapper.frame2second(end_frame_no)
                                clips.append(clip)
                                if end_frame_no-start_frame_no > 0:
                                    out_frames.append(end_frame_no-start_frame_no)
                            except Exception:
                                print(np.asarray(clip_skeleton).shape)
                                traceback.print_exc()

                            #print('{} ({}, {}), length: {}'.format(vid, start_frame_no, end_frame_no,end_frame_no-start_frame_no))
                        else:
                            pass
                            #print('{} ({}, {}) - consecutive missing frames'.format(vid, start_frame_no, end_frame_no))
                except Exception:
                    traceback.print_exc()

            if len(clips) > 0:
                with dataset.begin(write=True) as txn:
                    k = '{:010}'.format(v_i).encode('ascii')
                    out = {"vid": vid, "clips": clips}
                    v = pyarrow.serialize(out).to_buffer()
                    txn.put(k, v)
        dataset.sync()
        dataset.close()
    except Exception:
        traceback.print_exc()
    return out_frames, valid_clip_count


def make_ted_lmdb_gesture_dataset():
    n_saved_clips = [0, 0, 0]


    prefix = ""
    dataset_train = prefix+"/mnt/ssd-tb/gesture_data_video_entity/dataset_train"#lmdb.open("/mnt/98072f92-dbd5-4613-b3b6-f857bcdcdadc/temp/dataset_train",map_size=map_size)
    dataset_test = prefix+"/mnt/ssd-tb/gesture_data_video_entity/dataset_test"#lmdb.open("/mnt/98072f92-dbd5-4613-b3b6-f857bcdcdadc/temp/dataset_test",map_size=map_size)
    dataset_val = prefix+"/mnt/ssd-tb/gesture_data_video_entity/dataset_val"#lmdb.open("/mnt/98072f92-dbd5-4613-b3b6-f857bcdcdadc/temp/dataset_val",map_size=map_size)
    print(my_config.VIDEO_PATH + "/*.webm")
    video_files = list(sorted(glob.glob(my_config.VIDEO_PATH + "/*" + my_config.FILETYPE), key=os.path.getmtime))
    random.shuffle(video_files)
    print(video_files)
    tlist = []
    exec = ThreadPoolExecutor(max_workers=8)
    map_size = 1024 * 200  # in MB
    map_size <<= 20  # in B
    #dataset = lmdb.open(dataset_train, map_size=map_size)
    for v_i, video_file in enumerate(tqdm(video_files)):
        ra = random.randint(0,100)
        print(v_i)
        dataset = dataset_train
        print(video_file)

        tlist.append(exec.submit(run,v_i,video_file,dataset))
        #run(v_i, video_file, dataset)
    tbar = tqdm(tlist)
    ef = 0
    ret_frames = []
    ret_clips = []
    for t in tbar:
        try:
            frames,clips = t.result()
            ret_frames.extend(frames)
            ret_clips.append(clips)
            r1 = np.asarray(ret_frames)
            r2 = np.asarray(ret_clips)
            print("frames:",r1.min(),r1.max(),  np.median(r1),np.std(r1),r1.sum())
            print("clips:", r2.min(), r2.max(), np.median(r2),np.std(r2),r2.sum())

        except Exception:
            pass
            #traceback.print_exc()
    np.save("stat_frames.npy",r1)
    np.save("stat_clips.npy",r2)

    # for debugging
    # if vid == 'yq3TQoMjXTw':
    #     break
    #print('no. of saved clips: train {}, val {}, test {}'.format(n_saved_clips[0], n_saved_clips[1], n_saved_clips[2]))


if __name__ == '__main__':
    make_ted_lmdb_gesture_dataset()
