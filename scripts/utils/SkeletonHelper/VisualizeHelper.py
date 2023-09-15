import math

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# matplotlib.use('GTK3Agg')
# matplotlib.use('Agg')
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from scripts.utils.SkeletonHelper.HandStiffness import calculateAngleForHands, calculateHandStiffness, relaxedByFinger, \
    getHandFingerTipAnnotationPos
from scripts.utils.SkeletonHelper.helper_function import halpeToCocoSkeleton


def PersonlistToVisualizeSequenceData(personlist):
    """
    For this function I'm assuming that the personlist is not shuffled and a near direct output of the Realtimesequence
    class. If this becomes a problem in the future, the class has a "seen" list in the dict, with which one can sort the sequence.

    :type personlist: list
    list of frames from the RealtimeSequence class.
    """
    persondict = {}
    for frame in personlist:
        for frameP in frame.values():
            if frameP["id"] in persondict.keys():
                personout = persondict[frameP["id"]]
            else:
                personout = {"poses": [], "trajectorie": [], "frames": [], "keypoints": []}
            personout["poses"].append(frameP["current_3d_pose"])
            personout["trajectorie"].append(frameP["current_3d_traj"])
            personout["frames"].append(frameP["current_frame"])
            personout["keypoints"].append(frameP["current_2d_pose"])
            persondict[frameP["id"]] = personout
    for k in persondict.keys():
        person = persondict[k]
        person["poses"] = np.asarray(person["poses"])
        person["trajectorie"] = np.asarray(person["trajectorie"])
        person["frames"] = np.asarray(person["frames"])
        person["keypoints"] = np.asarray(person["keypoints"])
        persondict[k] = person
    return persondict


def resize2SquareKeepingAspectRation(img, size, interpolation):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w) / 2.)
    y_pos = int((dif - h) / 2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)


class VisualizeCvClass:
    skeleton_parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    big_skeleton_parents = [-1, 0, 0, 1, 2, 18, 18, 5, 6, 7, 8, 19, 19, 11, 12, 13, 14, 0, 0, 18, 24, 25, 24, 25, 15,
                            16]
    hand_parent = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

    def __init__(self, size=800, full136=True, videoframes=True):
        self.size = size
        self.radius = 2.5
        self.full136 = full136
        self.framecount = 0

        self.handL_parent = np.asarray(self.hand_parent) + 94
        self.handL_parent[0] = -1
        self.handR_parent = np.asarray(self.hand_parent) + 115
        self.handR_parent[0] = -1
        self.traj_last = None
        self.traj_history = []
        self.traj_color = []
        self.leftHandStiffnessHistory = []
        self.rightHandStiffnessHistory = []

    def reset(self):
        self.traj_last = None
        self.traj_history = []
        self.traj_color = []

    def visualizeRealtimeSequenceOutput(self, frameP, extraName=""):
        pose = frameP["current_3d_pose"]
        traj_out = frameP["current_3d_traj"]
        videoFrame = None
        if "current_frame" in frameP:
            videoFrame = frameP["current_frame"]
        keypoint = frameP["current_2d_pose"]
        scores = frameP["current_2d_score"].flatten()
        self.visualizeOneFrame(pose, traj_out, videoFrame, keypoint, scores, extraName=extraName)

    def rotateSkeleton(self, pose, degree=90, axis=(0, 1, 0)):
        rotation_radians = np.radians(degree)
        rotation_axis = np.array(axis)
        rotation_vector = rotation_radians * rotation_axis
        rotation = Rotation.from_rotvec(rotation_vector)
        origin = pose[8]
        pose[:] -= origin
        rotated_vec = rotation.apply(pose)
        rotated_vec += origin
        return rotated_vec

    r"""
        Calculates Angle for the hand joints.
        A is the anchor point (wrist)
        B is the lower hand joint
        C is the upper hand joint
        We create a direction vector from A and B 
        and then calculate the angle for C
    """

    def visualizeOneFrame(self, pose, traj_out, videoFrame, keypoint, scores=None, score_threshold=0.1,
                          render_traj=False, wait=False, extraName=""):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 0.6
        fontColor = (255, 255, 255)
        lineType = 1
        scores_coco = None
        if scores is not None:
            scores_coco = halpeToCocoSkeleton(scores.reshape((136, 1))).flatten()

        pose = pose.copy()

        org_size = None
        if videoFrame is not None:
            videoFrame = videoFrame.copy()
            org_size = videoFrame.shape
            keypoint = keypoint.copy()
            keypoint = keypoint * [org_size[1], org_size[0]]
            for i in range(len(self.big_skeleton_parents)):
                intensity = 1 if scores is None else max(score_threshold, scores[i])
                j_parent = self.big_skeleton_parents[i]
                if j_parent == -1:
                    continue
                if intensity > score_threshold:
                    cv2.line(videoFrame,
                             (int(keypoint[i, 0]), int(keypoint[i, 1])),
                             (int(keypoint[j_parent, 0]), int(keypoint[j_parent, 1])), (0, 255 * intensity, 0),
                             thickness=3, lineType=8)
                    cv2.putText(videoFrame, str(i),
                                (int(keypoint[i, 0]), int(keypoint[i, 1])),
                                font,
                                fontScale,
                                fontColor,
                                lineType)
            if keypoint.shape[0] == 136:

                for i in range(26, 94):
                    intensity = 1 if scores is None else max(score_threshold, scores[i])
                    if intensity > score_threshold:
                        cv2.circle(videoFrame, (int(keypoint[i, 0]), int(keypoint[i, 1])), radius=2,
                                   color=(0, 0, 255 * intensity), thickness=-1)
                for i in range(94, 115):
                    intensity = 1 if scores is None else max(score_threshold, scores[i])
                    j_parent = self.handL_parent[i - 94]
                    if j_parent == -1:
                        continue
                    if intensity > score_threshold:
                        cv2.line(videoFrame,
                                 (int(keypoint[i, 0]), int(keypoint[i, 1])),
                                 (int(keypoint[j_parent, 0]), int(keypoint[j_parent, 1])), (255 * intensity, 0, 0),
                                 thickness=3, lineType=8)
                for i in range(115, 136):
                    intensity = 1 if scores is None else max(score_threshold, scores[i])
                    j_parent = self.handR_parent[i - 115]
                    if j_parent == -1:
                        continue
                    if intensity > score_threshold:
                        cv2.line(videoFrame,
                                 (int(keypoint[i, 0]), int(keypoint[i, 1])),
                                 (int(keypoint[j_parent, 0]), int(keypoint[j_parent, 1])),
                                 (255 * intensity, 255 * intensity, 0),
                                 thickness=3, lineType=8)

            videoFrame = resize2SquareKeepingAspectRation(videoFrame, self.size, cv2.INTER_AREA)
            # cv2.imshow(extraName+" videoframe",cv2.cvtColor(videoFrame, cv2.COLOR_BGR2RGB))
            cv2.imshow(extraName + " videoframe", videoFrame)
            # cv2.imwrite("tempICMI/" + str(self.framecount).zfill(5)+"_out.png")
            self.framecount += 1

        # for i in range(3):
        #    print(pose[:,i].mean(),pose[:,i].min(),pose[:,i].max())
        # print("------")
        pose /= 1.5
        pose += [1.5, 0.4, 0]

        # pose[:, 0] -= pose[:, 0].min()
        # pose[:,0]  /= pose[:,0].max()
        # pose[:, 1] -= pose[:, 1].min()
        # pose[:, 1] /= pose[:, 1].max()
        # pose[:, 2] /= 4
        # pose[:,:2] /= 10
        # pose[:, :2] += 0.4

        vis_image = np.zeros((int(self.size), int(self.size * 3), 3))

        assert pose.shape[0] == 53
        # assert points.txt[:, :2].max() <= 1.00

        pose1 = self.rotateSkeleton(pose.copy(), -90, (0, 1, 0))
        pose2 = self.rotateSkeleton(pose.copy(), 90, (0, 1, 0))
        skeleton_parents = np.asarray([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        hand_parents = np.asarray([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
        hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
        hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
        hand_parents_l = hand_parents_l + 17
        hand_parents_r = hand_parents_r + 17 + 21
        if scores_coco is None:
            scores_coco = np.ones((59,))
        skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)
        scores_coco = np.asarray(scores_coco)
        scores_coco = np.concatenate((scores_coco, np.ones((21,)) * scores_coco[13], np.ones((21,)) * scores_coco[16]))

        lfingerAngle, rfingerAngle = calculateAngleForHands(pose)
        lfingerStiffness, rfingerStiffness = calculateHandStiffness(lfingerAngle, rfingerAngle)
        lfingerStiffness = relaxedByFinger(lfingerStiffness)
        rfingerStiffness = relaxedByFinger(rfingerStiffness)
        ls = "Left Hand Stiffness:  {0:+.5f}".format(np.asarray(lfingerStiffness).mean())
        rs = "Right Hand Stiffness: {0:+.5f}".format(np.asarray(rfingerStiffness).mean())

        self.leftHandStiffnessHistory.append(np.asarray(lfingerStiffness).mean())
        self.rightHandStiffnessHistory.append(np.asarray(rfingerStiffness).mean())
        self.leftHandStiffnessHistory = self.leftHandStiffnessHistory[-20:]
        self.rightHandStiffnessHistory = self.rightHandStiffnessHistory[-20:]
        if len(self.leftHandStiffnessHistory) == 20:
            lchange = "Left Stiffness change:  {0:+.5f}".format(
                savgol_filter(np.asarray(self.leftHandStiffnessHistory).flatten(), window_length=7, polyorder=3,
                              deriv=1).flatten().mean())
            rchange = "Right Stiffness change: {0:+.5f}".format(
                savgol_filter(np.asarray(self.leftHandStiffnessHistory).flatten(), window_length=7, polyorder=3,
                              deriv=1).flatten().mean())

        # print("Right Hand Stiffness:", np.asarray(rfingerStiffness).mean())
        # print("-----------")

        lAnnoP, rAnnoP = getHandFingerTipAnnotationPos(pose)

        # nicePrintHandTable(lfingerAngle, rfingerAngle)

        for i, j_parent in enumerate(skeleton_parents):
            if j_parent == -1:
                continue
            if scores is None or scores_coco[i] > score_threshold:
                cv2.line(vis_image, (int(pose[i, 0] * self.size), int(pose[i, 1] * self.size)),
                         (int(pose[j_parent, 0] * self.size), int(pose[j_parent, 1] * self.size)), (0, 255, 0),
                         thickness=1,
                         lineType=8)
                cv2.line(vis_image, (int((pose1[i, 0] + 1) * self.size), int((pose1[i, 1]) * self.size)),
                         (int((pose1[j_parent, 0] + 1) * self.size), int((pose1[j_parent, 1]) * self.size)),
                         (0, 255, 0),
                         thickness=1,
                         lineType=8)
                cv2.line(vis_image, (int((pose2[i, 0] - 1) * self.size), int((pose2[i, 1]) * self.size)),
                         (int((pose2[j_parent, 0] - 1) * self.size), int((pose2[j_parent, 1]) * self.size)),
                         (0, 255, 0),
                         thickness=1,
                         lineType=8)

                # cv2.putText(vis_image, str(i),
                #             (int((pose1[i, 0] + 0.25) * self.size), int(pose1[i, 1] * self.size)),
                #             font,
                #             fontScale,
                #             fontColor,
                #             lineType)
                # cv2.putText(vis_image, str(i),
                #             (int((pose2[i, 0] - 0.25) * self.size), int(pose2[i, 1] * self.size)),
                #             font,
                #             fontScale,
                #             fontColor,
                #             lineType)
        for i in range(len(lAnnoP)):
            AP = lAnnoP[i]
            AN = lfingerStiffness[i]
            if not math.isnan(AN):
                cv2.putText(vis_image, str(round(AN)),
                            (int(AP[0] * self.size), int(AP[1] * self.size)),
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            AP = rAnnoP[i]
            AN = rfingerStiffness[i]
            if not math.isnan(AN):
                cv2.putText(vis_image, str(round(AN)),
                            (int(AP[0] * self.size), int(AP[1] * self.size)),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
        cv2.putText(vis_image, ls,
                    (0, 20),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(vis_image, rs,
                    (0, 40),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        if len(self.leftHandStiffnessHistory) == 20:
            cv2.putText(vis_image, lchange,
                        (0, 60),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.putText(vis_image, rchange,
                        (0, 80),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        cv2.imshow(extraName + " skeleton vis", vis_image)

        if render_traj:
            traj_image = np.zeros((int(self.size), int(self.size), 3))

            random_color = np.random.uniform(0, 1, (3,)).flatten()
            if self.traj_last is None:
                cv2.circle(videoFrame, (int(self.size / 2), int(self.size / 2)), radius=2, color=(0, 0, 255),
                           thickness=-1)
                self.traj_history.append((0, 0))
                self.traj_color.append(random_color)
            else:
                traj_dif = (traj_out[0] - self.traj_last[0], traj_out[1] - self.traj_last[1])
                traj_dif += (self.traj_history[-1][0], self.traj_history[-1][1])
                self.traj_history.append(traj_dif)
                self.traj_color.append(random_color)
                for i, tt in enumerate(self.traj_history):
                    if i == 0:
                        continue
                    t2 = self.traj_history[i - 1]
                    cv2.line(traj_image, (int((tt[0] + 0.5) * self.size), int((tt[1] + 0.5) * self.size)),
                             (int((t2[0] + 0.5) * self.size), (int((t2[1] + 0.5) * self.size))),
                             self.traj_color[i].tolist(), thickness=1, lineType=8)
            self.traj_history = self.traj_history[-100:]
            self.traj_color = self.traj_color[-100:]
            self.traj_last = traj_out
            cv2.imshow(extraName + " trajectory map", traj_image)

    def VisUpperBody(self, pose, scores, score_threshold=0.05):
        pose = pose.copy()

        pose *= 2.5

        scores = scores.copy()
        # pose += 0.
        # #print(pose.max(axis=0).max())
        pose[:, 0] -= pose[2, 0]
        pose[:, 1] -= pose[2, 1]
        # pose /= pose.max(axis=0).max()
        #
        scores_coco = scores
        # pose /= 3

        # pose1 = self.rotateSkeleton(pose.copy(), -90, (0, 1, 0))
        # pose2 = self.rotateSkeleton(pose.copy(), 90, (0, 1, 0))

        pose += [0.5, 0.4, 0]
        size = self.size
        vis_image = np.zeros((int(self.size), int(self.size), 3))
        assert pose.shape[0] == 53
        # assert points.txt[:, :2].max() <= 1.00

        ubs = 6

        skeleton_parents = np.asarray(
            [-1, 0, 7 - ubs, 8 - ubs, 9 - ubs, 8 - ubs, 11 - ubs, 12 - ubs, 8 - ubs, 14 - ubs, 15 - ubs])
        hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, -4, 13, 14, 15, -4, 17, 18, 19])
        hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, -22, 13, 14, 15, -22, 17, 18, 19])
        hand_parents_l = hand_parents_l + 17 - ubs
        hand_parents_r = hand_parents_r + 17 + 21 - ubs

        skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)

        scores_coco = np.asarray(scores_coco)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 0.6
        fontColor = (255, 255, 255)
        lineType = 1
        # scores_coco = np.ones((53,))
        for i, j_parent in enumerate(skeleton_parents):
            if j_parent == -1:
                continue
            if scores_coco[i] > score_threshold:
                cv2.line(vis_image, (int(pose[i, 0] * size), int(pose[i, 1] * size)),
                         (int(pose[j_parent, 0] * size), int(pose[j_parent, 1] * size)), (255, 255, 255),
                         thickness=3,
                         lineType=8)
                # cv2.putText(vis_image, str(i),
                #             (int(pose[i, 0] * size), int(pose[i, 1] * size)),
                #             font,
                #             fontScale,
                #             fontColor,
                #             lineType)
        # cv2.imshow(" skeleton vis:"+name, vis_image)

        # os.makedirs("vis_out/"+name+"/",exist_ok=True)
        # cv2.imwrite("vis_out/"+name+"/"+"{:06d}".format(ie)+".png",vis_image)
        return vis_image

    def VisSkeletonArmGestures(self, ine, translateArms=True):
        pose = ine[0]
        # pose += 0.
        # #print(pose.max(axis=0).max())
        # pose -= pose[8]
        # pose /= pose.max(axis=0).max()
        #
        scores_coco = ine[1]
        pose /= 3
        pose += [0.5, 0.2, 0]
        if translateArms:
            pose[0:25] += [0.05, 0, 0]
            pose[24:] -= [0.05, 0, 0]
        score_threshold = 0.01
        size = self.size
        vis_image = np.zeros((int(self.size // 2), int(self.size), 3))
        assert pose.shape[0] == 48
        # assert points.txt[:, :2].max() <= 1.00

        pose1 = self.rotateSkeleton(pose.copy(), -90, (0, 1, 0))
        pose2 = self.rotateSkeleton(pose.copy(), 90, (0, 1, 0))

        ubs = 6

        skeleton_parents = np.asarray(
            [-1, 0, 7 - ubs, 8 - ubs, 9 - ubs, 8 - ubs, 11 - ubs, 12 - ubs, 8 - ubs, 14 - ubs, 15 - ubs])
        hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
        hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
        hand_parents_l = hand_parents_l + 17 - ubs
        hand_parents_r = hand_parents_r + 17 + 21 - ubs

        skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)

        scores_coco = np.asarray(scores_coco)
        scores_coco = np.ones((42,))
        for i, j_parent in enumerate(skeleton_parents):
            if j_parent == -1:
                continue
            if scores_coco[i] > score_threshold:
                cv2.line(vis_image, (int(pose[i, 0] * size), int(pose[i, 1] * size)),
                         (int(pose[j_parent, 0] * size), int(pose[j_parent, 1] * size)), (0, 255, 0),
                         thickness=1,
                         lineType=8)
                cv2.line(vis_image, (int((pose1[i, 0] + 0.35) * size), int((pose1[i, 1]) * size)),
                         (int((pose1[j_parent, 0] + 0.35) * size), int((pose1[j_parent, 1]) * size)), (0, 255, 0),
                         thickness=1,
                         lineType=8)
                cv2.line(vis_image, (int((pose2[i, 0] - 0.35) * size), int((pose2[i, 1]) * size)),
                         (int((pose2[j_parent, 0] - 0.35) * size), int((pose2[j_parent, 1]) * size)),
                         (0, 255, 0),
                         thickness=1,
                         lineType=8)
        # cv2.imshow(" skeleton vis:"+name, vis_image)

        # os.makedirs("vis_out/"+name+"/",exist_ok=True)
        # cv2.imwrite("vis_out/"+name+"/"+"{:06d}".format(ie)+".png",vis_image)
        return vis_image
