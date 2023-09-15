import os
from collections import Counter

import cv2
import pympi
import bz2

import librosa
import numpy as np
import pickle

from tqdm import tqdm

from helper_function import annotationsToTimings, getHandPositions, getAllAnnotationInRange, getAnnotationwithTiming, getEntityFromWordList, getAnnotationFromTime, getPositionFromWordList

__VERSION__ = 0.1


class RangeKeyDict:
    def __init__(self, my_dict):
        # !any(!A or !B) is faster than all(A and B)
        assert not any(map(lambda x: not isinstance(x, tuple) or len(x) != 2 or x[0] > x[1], my_dict))

        def lte(bound):
            return lambda x: bound <= x

        def gt(bound):
            return lambda x: x < bound

        # generate the inner dict with tuple key like (lambda x: 0 <= x, lambda x: x < 100)
        self._my_dict = {(lte(k[0]), gt(k[1])): v for k, v in my_dict.items()}

    def __getitem__(self, number):
        from functools import reduce
        _my_dict = self._my_dict
        try:
            result = next((_my_dict[key] for key in _my_dict if list(reduce(lambda s, f: filter(f, s), key, [number]))))
        except StopIteration:
            raise KeyError(number)
        return result

    def get(self, number, default=None):
        try:
            return self.__getitem__(number)
        except KeyError:
            return default


if __name__ == '__main__':
    range_key_dict = RangeKeyDict({
        (0, 100): 'A',
        (100, 200): 'B',
        (200, 300): 'C',
    })

    # test normal case
    assert range_key_dict[70] == 'A'
    assert range_key_dict[170] == 'B'
    assert range_key_dict[270] == 'C'

    # test case when the number is float
    assert range_key_dict[70.5] == 'A'

    # test case not in the range, with default value
    assert range_key_dict.get(1000, 'D') == 'D'


def list_to_range_dict(l1):
    ret = {}
    for lout in l1:
        ret[(lout[0],lout[1])] = lout[2]
    return ret

def get(d,s):
    try:
        for k,v in d.items():
            if k[0] <= s <= k[1]:
                return v
    except Exception:
        pass
    return None

def removeValue(l1,val=''):
    ret = []
    for lin in l1:
        if lin[2] != val:
            ret.append(lin)
    return ret


class SaGA_dataloader():

    def __init__(self, video_file,returnVideoFrames=False,returnAudioFrames=False):
        """
        The class expects the following convention of filenames to work correctly. (this is already given in the SaGA1 dataset)
        <num> = study file number
        video file: V<num>K<x>.mov.mp4
        pose estimation file: V<num>K<x>.mov.mp4_calc.kp
        scene_file: V<num>K<x>.mov.mp4_scene.pk
        elan file: <num>_video.eaf

        :param video_file: the video file that should be loaded. All other files will be inferred from the filename.
        """
        self.video_file = video_file

        self.pose_estimation = video_file+"_calc.kp"
        self.scene_file = video_file + "_scene.pk"

        self.elanFile = os.path.splitext(self.video_file)[0]+".eaf"
        # elanFile = os.path.basename(self.pose_estimation)
        # elanFile = elanFile.split("V")[1].split("K")[0]
        # self.elanFile = os.path.dirname(os.path.abspath(self.video_file))+"/"+"{:02d}".format(int(elanFile)) + "_video.eaf"
        # print(self.elanFile)
        self.english_sub_file = os.path.dirname(os.path.abspath(self.video_file))+"/english/"+os.path.basename(self.elanFile) + ".txt_converted.txt"

        assert os.path.exists(self.video_file)
        assert os.path.exists(self.pose_estimation)
        #assert os.path.exists(self.scene_file)
        assert os.path.exists(self.elanFile)

        self.return_video_frames = returnVideoFrames
        self.return_audio_frames = returnAudioFrames

        self.init_data()

        self.current_frame = 0

    def getcurrentFrame(self):
        return self.current_frame

    def init_data(self):

        self.current_frame = 0
        video = cv2.VideoCapture(self.video_file)
        self.duration = video.get(cv2.CAP_PROP_POS_MSEC)
        self.frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        if not self.return_video_frames:
            video.release()
            del video
        else:
            self.video_cap = video

        if self.return_audio_frames:
            audio, samplingrate = librosa.load(self.video_file, sr=16000)
            fps_samplerate = int(16000//self.fps)
            if audio.shape[0] % fps_samplerate != 0:
                audio = np.concatenate((audio, np.zeros((fps_samplerate - audio.shape[0] % fps_samplerate,))))
            self.audio = audio.reshape(-1, fps_samplerate)

        ee = pympi.Elan.Eaf(self.elanFile)

        left,right = getHandPositions(ee)
        self.left_hand_data = [list_to_range_dict(l1) for l1 in left]
        self.right_hand_data = [list_to_range_dict(l1) for l1 in right]

        R_S_Form = getAnnotationwithTiming(ee, "R.S.Form")

        self.R_S_Semantic_Feature = getAnnotationwithTiming(ee,"R.S.Semantic Feature" if "R.S.Semantic Feature" in ee.tiers.keys() else "R.S.Sematic Feature")
        self.R_G_Left_Semantic = getAnnotationwithTiming(ee,"R.G.Left Semantic" if "R.G.Left Semantic" in ee.tiers.keys() else "R.G.Left Semactic" if "R.G.Left Semactic" in ee.tiers.keys() else "R.G.Left Semantic ")
        self.R_G_Right_Semantic = getAnnotationwithTiming(ee,"R.G.Right Semantic" if "R.G.Right Semantic" in ee.tiers.keys() else "R.G.Right Semantic ")

        entity_word = {}
        for r_s in self.R_S_Semantic_Feature:
            cur_words = getAllAnnotationInRange(R_S_Form, r_s[0], r_s[1], extraTime=500)
            entity = getEntityFromWordList(cur_words)
            #if entity is not None:
            entity_word[(r_s[0],r_s[1])] = entity


        def doSemantic(sem):
            ret = {}
            for r_s in sem:
                cur_words = getAllAnnotationInRange(R_S_Form, r_s[0], r_s[1], extraTime=1000)
                position = getPositionFromWordList(cur_words)
                if position is not None:
                    ret[(r_s[0], r_s[1])] = position
            return ret

        self.R_G_Left_Semantic_Position = doSemantic(self.R_G_Left_Semantic)
        self.R_G_Right_Semantic_Position = doSemantic(self.R_G_Right_Semantic)
        self.R_G_Speech_Semantic_Position = doSemantic(self.R_S_Semantic_Feature)


        self.R_S_Semantic_Feature = list_to_range_dict(self.R_S_Semantic_Feature)
        self.R_G_Left_Semantic = list_to_range_dict(self.R_G_Left_Semantic)
        self.R_G_Right_Semantic = list_to_range_dict(self.R_G_Right_Semantic)

        self.entity_word = entity_word
        self.left_phase, self.right_phase = list_to_range_dict(getAnnotationwithTiming(ee, "R.G.Left.Phase")), list_to_range_dict(getAnnotationwithTiming(ee,"R.G.Right.Phase"))

        self.R_S_Form = list_to_range_dict(removeValue(getAnnotationwithTiming(ee, "R.S.Form")))
        self.F_S_Form = list_to_range_dict(removeValue(getAnnotationwithTiming(ee, "F.S.Form")))

        self.R_G_Right_Phrase = list_to_range_dict(removeValue(getAnnotationwithTiming(ee, "R.G.Right.Phrase")))
        self.R_G_Left_Phrase = list_to_range_dict(removeValue(getAnnotationwithTiming(ee, "R.G.Left.Phrase")))

        self.F_G_Right_Phrase = list_to_range_dict(removeValue(getAnnotationwithTiming(ee, "F.G.Right.Phrase")))
        self.F_G_Left_Phrase = list_to_range_dict(removeValue(getAnnotationwithTiming(ee, "F.G.Left.Phrase")))

        self.gesture_data = pickle.load(bz2.BZ2File(self.pose_estimation, "rb"))




    def __len__(self):
        le = min(self.frame_count,len(self.gesture_data))
        if self.return_audio_frames:
            le = min(le,len(self.audio))
        return le

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame > self.__len__():
            raise StopIteration
        if self.return_audio_frames:
            if self.current_frame >= len(self.audio):
                raise StopIteration
        if self.current_frame >= len(self.gesture_data):
            raise StopIteration

        current_time = int(self.current_frame * (1000/self.fps))
        ret = {}

        if self.return_audio_frames:
            # audio data for the current frame
            ret["Audio"] = self.audio[self.current_frame]

        # speech transcript word of the current frame
        ret["R_S_Text"] = get(self.R_S_Form,current_time)
        ret["F_S_Text"] = get(self.F_S_Form,current_time)

        # semantic annotation for the current gesture
        ret["R_G_Left_Semantic"] = get(self.R_G_Left_Semantic,current_time)
        ret["R_G_Right_Semantic"] = get(self.R_G_Right_Semantic,current_time)

        # speech semantic feature
        ret["R_S_Semantic"] = get(self.R_S_Semantic_Feature,current_time)

        # entity from the semantic features clustered by discussed topic in the text
        ret["R_S_Semantic_Entity"] = get(self.entity_word,current_time)

        ret["R_G_Left_Semantic_Position"] = get(self.R_G_Left_Semantic_Position,current_time)
        ret["R_G_Right_Semantic_Position"] = get(self.R_G_Right_Semantic_Position, current_time)
        ret["R_S_Speech_Semantic_Position"] = get(self.R_G_Speech_Semantic_Position, current_time)

        # current interaction phase (prep,stroke,retr,post.hold)
        ret["R_G_Left_Phase"] = get(self.left_phase,current_time)
        ret["R_G_Right_Phase"] = get(self.right_phase,current_time)

        # All of the Left Hand Annotations
        ret["LeftHandShape"] = get(self.left_hand_data[0],current_time)
        ret["LeftPalmDirection"] = get(self.left_hand_data[1],current_time)
        ret["LeftBackOfHandDirection"] = get(self.left_hand_data[2],current_time)
        ret["LeftBackOfHandDirectionMovement"] = get(self.left_hand_data[3],current_time)
        ret["LeftWristPosition"] = get(self.left_hand_data[4],current_time)
        ret["LeftWristDistance"] = get(self.left_hand_data[5],current_time)
        ret["LeftPathOfWristLocation"] = get(self.left_hand_data[6],current_time)
        ret["LeftWristLocationMovementDirection"] = get(self.left_hand_data[7],current_time)
        ret["LeftExtent"] = get(self.left_hand_data[8],current_time)
        ret["LeftPractice"] = get(self.left_hand_data[9],current_time)

        # All of the Right Hand Annotations
        ret["RightHandShape"] = get(self.right_hand_data[0],current_time)
        ret["RightPalmDirection"] = get(self.right_hand_data[1],current_time)
        ret["RightBackOfHandDirection"] = get(self.right_hand_data[2],current_time)
        ret["RightBackOfHandDirectionMovement"] = get(self.right_hand_data[3],current_time)
        ret["RightWristPosition"] = get(self.right_hand_data[4],current_time)
        ret["RightWristDistance"] = get(self.right_hand_data[5],current_time)
        ret["RightPathOfWristLocation"] = get(self.right_hand_data[6],current_time)
        ret["RightWristLocationMovementDirection"] = get(self.right_hand_data[7],current_time)
        ret["RightExtent"] = get(self.right_hand_data[8],current_time)
        ret["RightPractice"] = get(self.right_hand_data[9],current_time)

        # R_G Gesture Phrases
        ret["R_G_Left_Phrase"] = get(self.R_G_Left_Phrase,current_time)
        ret["R_G_Right_Phrase"] = get(self.R_G_Right_Phrase,current_time)

        # F_G Gesture Phrases
        ret["F_G_Left_Phrase"] = get(self.F_G_Left_Phrase,current_time)
        ret["F_G_Right_Phrase"] = get(self.F_G_Right_Phrase,current_time)

        ret["gesture_data"] = self.gesture_data[self.current_frame]

        if self.return_video_frames:
            flag, frame = self.video_cap.read()
            if not flag:
                ret["Video_Frame"] = None
            else:
                ret["Video_Frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.current_frame += 1
        return ret




if __name__ == '__main__':
    import traceback
    folder = "/mnt/98072f92-dbd5-4613-b3b6-f857bcdcdadc/Owncloud/data/video/"
    for f in os.listdir(folder):
        if f and f.endswith(".mp4"):
            try:
                f = folder + f
                loader = SaGA_dataloader(f,returnVideoFrames=False,returnAudioFrames=False)
                for f in tqdm(loader):
                    if f["R_S_Speech_Semantic_Position"] is not None:
                        print("pos speech",f["R_S_Speech_Semantic_Position"])
                    if f["R_G_Left_Semantic_Position"] is not None:
                        print("pos left",f["R_G_Left_Semantic_Position"])
                    if f["R_G_Right_Semantic_Position"] is not None:
                        print("pos right",f["R_G_Right_Semantic_Position"])

                    if f["R_S_Semantic_Entity"] is not None:
                        print("entity",f["R_S_Semantic_Entity"])


            except Exception:
                print(f)
                traceback.print_exc()
            print("----")































