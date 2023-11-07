# ------------------------------------------------------------------------------
# Copyright 2019 ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

import copy
from data_utils import *
import numpy as np

from alphapose_helper_function import halpeToCocoSkeleton


class MainSpeakerSelector:
    def __init__(self, raw_skeleton_chunk):
        self.raw = raw_skeleton_chunk
        self.main_speaker_skeletons = self.find_main_speaker_skeletons(raw_skeleton_chunk)

    def get(self):
        return self.main_speaker_skeletons

    def find_main_speaker_skeletons(self, raw_skeleton_chunk):
        tracked_skeletons = []
        selected_skeletons = []  # reference skeleton
        for raw_frame in raw_skeleton_chunk:  # frame
            tracked_person = raw_frame[list(raw_frame.keys())[0]] if len(list(raw_frame.keys())) > 0 else None
            tracked_person = None
            if raw_frame is not None:
                if len(raw_frame.keys()) == 1:
                    tracked_person = raw_frame[list(raw_frame.keys())[0]]
                else:
                    raw_frame = list(raw_frame.values())
                    if selected_skeletons == []:
                        # select a main speaker
                        confidence_list = []

                        for person in raw_frame:  # people
                            body = person['current_3d_pose']
                            confidence_scores = halpeToCocoSkeleton(person["current_2d_score"])

                            mean_confidence = 0
                            n_points = 0

                            # Calculate the average of confidences of each person
                            for i in range(len(confidence_scores)):  # upper-body only
                                confidence = confidence_scores[i]
                                n_points += 1
                                mean_confidence += confidence
                            if n_points > 0:
                                mean_confidence /= n_points
                            else:
                                mean_confidence = 0
                            confidence_list.append(mean_confidence)

                        # select main_speaker with the highest average of confidence
                        if len(confidence_list) > 0:
                            max_index = confidence_list.index(max(confidence_list))
                            selected_skeletons = get_skeleton_from_frame(raw_frame[max_index])

                    if selected_skeletons is not None:
                        # find the closest one to the selected main_speaker's skeleton
                        tracked_person = self.get_closest_skeleton(raw_frame, selected_skeletons)
                        if tracked_person is not None:
                            tracked_person = raw_frame[tracked_person]

            # save
            if tracked_person is not None:
                selected_skeletons = get_skeleton_from_frame(tracked_person)

            skeleton_data = self.rework_skeleton_data(tracked_person)
            tracked_skeletons.append(skeleton_data)

        tracked_skeletons = self.smoothskeletondata(tracked_skeletons)
        return tracked_skeletons

    def rework_skeleton_data(self,skeleton_data):
        skeleton_data = copy.deepcopy(skeleton_data)
        if skeleton_data is None:
            return {}
        skeleton = get_skeleton_from_frame(skeleton_data)
        if skeleton is None:
            return {}
        else:
            skeleton_data['current_3d_pose'] = skeleton
            return skeleton_data

    def smoothskeletondata(self, tracked_skeletons):
        norm_history = []

        for i in range(len(tracked_skeletons)):
            if len(list(tracked_skeletons[i].keys())) > 0:
                norm_history.append(tracked_skeletons[i]["current_3d_pose"])
                tracked_skeletons[i]["current_3d_pose"] = np.mean(np.asarray(norm_history), axis=0)
                norm_history = norm_history[-3:]
            else:
                norm_history = []

        return tracked_skeletons

    def get_closest_skeleton(self, frame, selected_body):
        """ find the closest one to the selected skeleton """
        min_diff = 10000000
        ret_i = -1
        if frame is None:
            return None
        for i,person in enumerate(frame):
            body = get_skeleton_from_frame(person)
            if body is not None:
                dist = np.linalg.norm(selected_body - body)
                if dist < min_diff:
                    min_diff = dist
                    ret_i = i
        if len(frame) == 0:
            return None
        if ret_i == -1:
            return None
        return ret_i

