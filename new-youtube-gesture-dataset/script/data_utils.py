# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------
import bz2
import glob
import os.path

import matplotlib
import cv2
import re
import json
import _pickle as pickle
from webvtt import WebVTT
from config import my_config


###############################################################################
# SKELETON
from SkeletonHelper.helper_function import checkifwordtranscript, convertTime, removeRotation, normalizePosition, \
    onlyGestures, halpeTo59PoseSkeleton, onlyUpperBodyScores, onlyUpperBodyPoses, deboneUpper, rescaleUpperoneLengths, \
    reboneUpper, calculateUpperBodyScoreThreshold, checkHandtracked


def draw_skeleton_on_image(img, skeleton, thickness=15):
    if not skeleton:
        return img

    new_img = img.copy()
    for pair in SkeletonWrapper.skeleton_line_pairs:
        pt1 = (int(skeleton[pair[0] * 3]), int(skeleton[pair[0] * 3 + 1]))
        pt2 = (int(skeleton[pair[1] * 3]), int(skeleton[pair[1] * 3 + 1]))
        if pt1[0] == 0 or pt2[1] == 0:
            pass
        else:
            rgb = [v * 255 for v in matplotlib.colors.to_rgba(pair[2])][:3]
            cv2.line(new_img, pt1, pt2, color=rgb[::-1], thickness=thickness)

    return new_img


def is_list_empty(my_list):
    return all(map(is_list_empty, my_list)) if isinstance(my_list, dict) else False


def get_closest_skeleton(frame, selected_body):
    """ find the closest one to the selected skeleton """
    diff_idx = [i * 3 for i in range(8)] + [i * 3 + 1 for i in range(8)]  # upper-body

    min_diff = 10000000
    tracked_person = None
    for person in frame:  # people
        body = get_skeleton_from_frame(person)

        diff = 0
        n_diff = 0
        for i in diff_idx:
            if body[i] > 0 and selected_body[i] > 0:
                diff += abs(body[i] - selected_body[i])
                n_diff += 1
        if n_diff > 0:
            diff /= n_diff
        if diff < min_diff:
            min_diff = diff
            tracked_person = person

    base_distance = max(abs(selected_body[0 * 3 + 1] - selected_body[1 * 3 + 1]) * 3,
                        abs(selected_body[2 * 3] - selected_body[5 * 3]) * 2)
    if tracked_person and min_diff > base_distance:  # tracking failed
        tracked_person = None

    return tracked_person


def get_skeleton_from_frame(frame,checkHands=False):
    if 'current_3d_pose' in frame:
        pose = frame["current_3d_pose"].copy()
        score = frame["current_2d_score"].copy()
        score = halpeTo59PoseSkeleton(score)
        score = onlyUpperBodyScores(score)

        #pose = removeRotation(pose)
        pose = normalizePosition(pose)

        pose = onlyUpperBodyPoses(pose)

        pose = deboneUpper(pose)

        deboned_upper = pose.copy()

        pose = rescaleUpperoneLengths(pose)
        pose = reboneUpper(pose)

        if checkHands:
            if checkHandtracked(deboned_upper):
                return pose
        else:
            return pose
    return None


class SkeletonWrapper:
    # color names: https://matplotlib.org/mpl_examples/color/named_colors.png
    visualization_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'), (5, 6, 'g'),
                                (6, 7, 'lightgreen'),
                                (1, 8, 'darkcyan'), (8, 9, 'c'), (9, 10, 'skyblue'), (1, 11, 'deeppink'), (11, 12, 'hotpink'), (12, 13, 'lightpink')]
    skeletons = []
    skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'gold'), (1, 5, 'darkgreen'),
                           (5, 6, 'g'), (6, 7, 'lightgreen')]

    def __init__(self, basepath, vid):
        # load skeleton data (and save it to pickle for next load)
        pickle_file = glob.glob(basepath + '/' + vid + '.pickle')

        if pickle_file:
            try:
                with open(pickle_file[0], 'rb') as file:
                    self.skeletons = pickle.load(file)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(file)
        else:
            files = glob.glob(basepath + '/' + vid + '/*.json')
            if len(files) > 10:
                files = sorted(files)
                self.skeletons = []
                for file in files:
                    self.skeletons.append(self.read_skeleton_json(file))
                with open(basepath + '/' + vid + '.pickle', 'wb') as file:
                    pickle.dump(self.skeletons, file)
            else:
                self.skeletons = []


    def read_skeleton_json(self, file):
        with open(file) as json_file:
            skeleton_json = json.load(json_file)
            return skeleton_json['people']


    def get(self, start_frame_no, end_frame_no, interval=1):

        chunk = self.skeletons[start_frame_no:end_frame_no]

        if is_list_empty(chunk):
            return []
        else: 
            if interval > 1:
                return chunk[::int(interval)]
            else:
                return chunk

class AlphaSkeletonWrapper:
    # color names: https://matplotlib.org/mpl_examples/color/named_colors.png
    skeletons = []


    def __init__(self, basepath, vid):
        # load skeleton data (and save it to pickle for next load)
        pickle_file = glob.glob(basepath + '/' + vid + '_calc.kp')

        if pickle_file:
            try:
                with open(pickle_file[0], 'rb') as file:
                    self.skeletons = pickle.load(bz2.BZ2File(file, "rb"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(file)
                raise e

    def getLen(self):
        return len(self.skeletons)

    def read_skeleton_json(self, file):
        return pickle.load(bz2.BZ2File(file, "rb"))


    def get(self, start_frame_no, end_frame_no, interval=1):

        chunk = self.skeletons[start_frame_no:end_frame_no]

        if is_list_empty(chunk):
            return []
        else:
            if interval > 1:
                return chunk[::int(interval)]
            else:
                return chunk


###############################################################################
# VIDEO
def read_video(base_path, vid):
    files = [base_path + '/' + vid]
    if len(files) == 0:
        return None
    elif len(files) >= 2:
        assert False
    filepath = files[0]

    video_obj = VideoWrapper(filepath)

    return video_obj


class VideoWrapper:
    video = []

    def __init__(self, filepath):
        self.filepath = filepath
        self.video = cv2.VideoCapture(filepath)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framerate = self.video.get(cv2.CAP_PROP_FPS)

    def get_video_reader(self):
        return self.video

    def frame2second(self, frame_no):
        return frame_no / self.framerate

    def second2frame(self, second):
        return int(round(second * self.framerate))

    def set_current_frame(self, cur_frame_no):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_no)


###############################################################################
# CLIP
def load_clip_data(vid):
    try:
        with open("{}/{}.json".format(my_config.CLIP_PATH, vid)) as data_file:
            data = json.load(data_file)
            return data
    except FileNotFoundError:
        print("{}/{}.json".format(my_config.CLIP_PATH, vid))
        return None


def load_clip_filtering_aux_info(vid):
    try:
        with open("{}/{}_aux_info.json".format(my_config.CLIP_PATH, vid)) as data_file:
            data = json.load(data_file)
            return data
    except FileNotFoundError:
        return None


#################################################################################
#SUBTITLE
class SubtitleWrapper:
    TIMESTAMP_PATTERN = re.compile('(\d+)?:?(\d{2}):(\d{2})[.,](\d{3})')

    def __init__(self, vid, mode,saga_path=None):
        self.subtitle = []
        if mode == 'auto':
            self.load_auto_subtitle_data(vid)
        elif mode == 'gentle':
            self.laod_gentle_subtitle(vid)
        elif mode == 'saga':
            self.load_saga_subtitle(saga_path)

    def get(self):
        return self.subtitle

    def load_saga_subtitle(self,path):
        word_list = []
        if path is None or not os.path.exists(path):
            print(path)
            print("no SaGA subtitle found. Searched for file:",path)
            return None
        with open(path) as f:
            lines = f.read().splitlines()
        for rline in lines:
            if len(rline) > 0:
                cpp = rline.split(";")
                if len(cpp) > 3:
                    if cpp[0] == "RS": # currently we are only interested in the speaker (the K3 in the video files)
                        print("yes")
                        start_time = float(cpp[1])/1000
                        end_time = float(cpp[2])/1000
                        # super simple splitting by duration and lettercount. Not super accurate, but better than nothing.
                        line = cpp[3]
                        num_letters = len(line)
                        duration = end_time - start_time
                        time_for_each_letter = duration / num_letters
                        words = line.split(" ")
                        acummulated_letters = 0
                        for i in range(len(words)):
                            word = words[i]
                            if i != 0:
                                acummulated_letters += 1  # +space

                            s1 = time_for_each_letter * acummulated_letters
                            acummulated_letters += len(word)  # +word
                            s2 = time_for_each_letter * acummulated_letters

                            word_info = {}
                            word_info['word'] = word
                            word_info['start'] = start_time + s1
                            word_info['end'] = start_time + s2
                            word_list.append(word_info)
        self.subtitle = word_list


    # using gentle lib
    def laod_gentle_subtitle(self,vid):
        try:
            with open("{}/{}_align_results.json".format(my_config.VIDEO_PATH, vid)) as data_file:
                data = json.load(data_file)
                if 'words' in data:
                    raw_subtitle = data['words']

                    for word in raw_subtitle :
                        if word['case'] == 'success':
                            self.subtitle.append(word)
                else:
                    self.subtitle = None
                return data
        except FileNotFoundError:
            self.subtitle = None

    # using youtube automatic subtitle
    def load_auto_subtitle_data(self, vid):
        lang = my_config.LANG
        postfix_in_filename = '.'+lang+'.vtt'
        #print(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)

        file_list = glob.glob(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)
        if len(file_list) > 1:
            print('more than one subtitle. check this.', file_list)
            self.subtitle = None
            assert False
        if len(file_list) == 1:
            vtt_file = list(WebVTT().read(file_list[0]))
            self.subtitle = convertVTT(vtt_file)
        else:
            print('subtitle file does not exist')
            print(my_config.SUBTITLE_PATH + '/*' + vid + postfix_in_filename)

            self.subtitle = None

    # convert timestamp to second
    def get_seconds(self, word_time_e):
        time_value = re.match(self.TIMESTAMP_PATTERN, word_time_e)
        if not time_value:
            print('wrong time stamp pattern')
            exit()

        values = list(map(lambda x: int(x) if x else 0, time_value.groups()))
        hours, minutes, seconds, milliseconds = values[0], values[1], values[2], values[3]

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def convertVTT(vtt_file):
    # two possibilities: only one word in two lines or sentence captioned with second texts
    # [[start_time,end_time,word],...]
    word_list = []
    history = []

    word_transcript = checkifwordtranscript(vtt_file)

    for cap in vtt_file:
        start_time = convertTime(cap.start)

        end_time = cap.end

        if word_transcript:
            # there is always only up to three two lines in a youtube subtitle. EXCEPT WHEN THERE ISN'T AND THERE IS MORE LINES!
            l1 = cap.lines[0].strip()
            l2 = cap.lines[1].strip() if len(cap.lines) > 1 else ''
            #check for the two possibilities
            if len(l1) > 0 and " " not in l1 and len(l2) == 0:
                #only one word with time
                word_info = {}
                word_info['word'] = l1
                word_info['start'] = start_time
                word_info['end'] = convertTime(end_time)
                word_list.append(word_info)
            elif ("<c>" in l1 or "</c>" in l1) or ("<c>" in l2 or "</c>" in l2):
                lparse = l1 if "<c>" in l1 or "</c>" in l1 else l2
                lparse = lparse.replace("<c>","").replace("</c>","") + "<"+str(end_time)+">"
                lwords = lparse.split(">")
                s_time = start_time
                for word_time in lwords:
                    if len(word_time) > 0:
                        word_time = word_time + ">"
                        word = word_time.split("<")[0].replace("\\","")
                        time = word_time.split("<")[1].split(">")[0]
                        time = convertTime(time)
                        word_info = {}
                        word_info['word'] = word
                        word_info['start'] = s_time
                        word_info['end'] = time
                        word_list.append(word_info)
                        s_time = time
                        history.append(word)
        else:
            end_time = convertTime(cap.end)
            #super simple splitting by duration and lettercount. Not super accurate, but better than nothing.
            line =  ' '.join(cap.lines)
            num_letters = len(line)
            duration = end_time-start_time
            time_for_each_letter = duration/num_letters
            words = line.split(" ")
            acummulated_letters = 0
            for i in range(len(words)):
                word = words[i]
                if i != 0:
                    acummulated_letters += 1 #+space

                s1 = time_for_each_letter*acummulated_letters
                acummulated_letters += len(word) #+word
                s2 = time_for_each_letter*acummulated_letters

                word_info = {}
                word_info['word'] = word
                word_info['start'] = start_time+s1
                word_info['end'] = start_time+s2
                word_list.append(word_info)

    return word_list

if __name__ == '__main__':
    #SubtitleWrapper("3D Rest  _ Deborah Patrick _ TEDxYouth@ATHS",mode="auto")
    SubtitleWrapper(vid=None,saga_path="/mnt/98072f92-dbd5-4613-b3b6-f857bcdcdadc/Owncloud/data/SaGA1/english/01_video.eaf_eng.txt_converted.txt",mode="saga")