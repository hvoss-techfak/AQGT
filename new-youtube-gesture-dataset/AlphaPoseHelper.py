# This is a sample Python script.
import timeit
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import types

import torch.multiprocessing

from Detectionloader_own import DetectionLoader_own, DetectionLoader_single

torch.multiprocessing.set_sharing_strategy('file_system')

#import cv2
import numpy as np
import torch
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord
from alphapose.utils.vis import vis_frame_fast as vis_frame
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
import mediapipe as mp
mp_hands = mp.solutions.hands
from common.model import TemporalModel
from trackers import track

args = types.SimpleNamespace()
args.checkpoint = "pretrained_models/halpe136_fast_res50_256x192.pth"
args.cfg = "configs/256x192_res50_lr1e-3_1x-simple.yaml"
args.detbatch = 5
args.posebatch = 60
args.detector = "yolo"
args.sp = False
args.flip = False
args.showbox = False
args.gpus = "0"

cfg = update_config(args.cfg)
args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.qsize = 10

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


class AlphaPoseHelper:
    def __init__(self, inputFile=None, justLoad=False,pose_track=True):
        args.pose_track = pose_track
        args.tracking = pose_track
        if not justLoad:
            if inputFile is None:
                self.det_loader = WebCamDetectionLoader(1, get_detector(args), cfg, args)
                self.frames = -1
            else:
                self.det_loader = DetectionLoader_single(inputFile, get_detector(args), cfg, args, batchSize=args.detbatch,
                                                  mode='video', queueSize=args.qsize)
                self.frames = -1
        self.loadPosemodel()
        self.batchSize = args.posebatch
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.cfg = cfg
        self.args = args
        self.load2Dto3Dmodel()
        self.maxHands = -1

    def __iter__(self):
        return self

    def __next__(self):
        try:
           bundle = next(self.det_loader)
           return self.doOneFrameFull(bundle)
        except StopIteration:
            raise StopIteration

    def getNumFrames(self):
        return self.frames

    def loadPosemodel(self):
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print('INFO: Loading pose model from %s...' % (args.checkpoint,))
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        print('INFO: Loading tracker...')
        self.tracker = Tracker(tcfg, args)
        if len(args.gpus) > 1:
            self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=args.gpus).to(args.device)
        else:
            self.pose_model.to(args.device)
        self.pose_model.eval()
        if torch.cuda.is_available():
            self.pose_model = self.pose_model.cuda()

        self.runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2)
        self.maxHands = 2

    def load2Dto3Dmodel(self):
        print('INFO: Loading 2D to 3D model...')
        self.model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25,
                                       channels=1024, dense=False)
        receptive_field = self.model_pos.receptive_field()
        print('INFO: Receptive field: {} frames'.format(receptive_field))
        self.pad = (receptive_field - 1) // 2  # Padding on each side
        if torch.cuda.is_available():
            self.model_pos = self.model_pos.cuda()
        #chk_filename = "pretrained_models/pretrained_h36m_detectron_coco.bin"
        chk_filename = "pretrained_models/pretrained_h36m_cpn.bin"
        print('INFO: Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)

        self.model_pos.load_state_dict(checkpoint['model_pos'])

        chk_filename = "pretrained_models/pretrained_243_h36m_detectron_coco_wtraj.bin"

        print('INFO: Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))

        print('INFO: Loading 2D to 3D trajectory model...')
        self.model_traj = TemporalModel(17, 2, 1, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25,
                                            channels=1024, dense=False)
        if torch.cuda.is_available():
            self.model_traj = self.model_traj.cuda()
        self.model_traj.load_state_dict(checkpoint['model_traj'])
        self.model_traj.eval()
        self.model_pos.eval()

    def doOneFrameHeat(self,bundle):
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = bundle
            if boxes is None:
                return boxes, scores, ids, None, cropped_boxes, orig_img, im_name
            # Pose Estimation
            inps = inps.to(args.device)
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % self.batchSize:
                leftover = 1
            num_batches = datalen // self.batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * self.batchSize:min((j + 1) * self.batchSize, datalen)]
                if args.flip:
                    inps_j = torch.cat((inps_j, flip(inps_j)))
                hm_j = self.pose_model(inps_j)
                if args.flip:
                    hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], self.pose_dataset.joint_pairs, shift=True)
                    hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                hm.append(hm_j)
            hm = torch.cat(hm)

            boxes, scores, ids, hm, cropped_boxes = track(self.tracker, args, orig_img, inps, boxes, hm,cropped_boxes,im_name, scores)
            hm = hm.cpu()
            # scores = np.asarray(scores)
            # hm = np.asarray(hm)
            # cropped_boxes = np.asarray(cropped_boxes)
            return boxes, scores, ids, hm, cropped_boxes, orig_img, im_name

    def do2Dto3D(self, pose_coords, w, h):
        import timeit
        pc_svt = normalize_screen_coordinates(pose_coords, w, h)
        pc_svt2 = np.expand_dims(np.pad(pc_svt, ((self.pad, self.pad), (0, 0), (0, 0)), 'edge'), axis=0)
        pc_svt2 = torch.from_numpy(pc_svt2.astype('float32'))
        pc_svt2 = pc_svt2.cuda()

        predicted_3d_pos = self.model_pos(pc_svt2)
        predicted_3d_pos = predicted_3d_pos.cpu().detach().numpy()
        predicted_3d_pos = predicted_3d_pos.reshape((-1, 17, 3))
        if args.pose_track:
            predicted_3d_traj = self.model_traj(pc_svt2)
            predicted_3d_traj = predicted_3d_traj.cpu().detach().numpy()
        else:
            predicted_3d_traj = np.zeros((predicted_3d_pos.shape[0],3))

        #ret = np.zeros((predicted_3d_pos.shape[0],17,3))
        #ret[:,:,:2] = pc_svt
        #ret[:,:,2] = predicted_3d_pos[:,:,2]
        return predicted_3d_pos,predicted_3d_traj.reshape((-1, 3))

    def visualize(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # image channel RGB->BGR
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        # final_result = []
        if boxes is None or len(boxes) == 0:
            return orig_img
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            # pred = hm_data.cpu().data.numpy()

            self.eval_joints = [*range(0, 136)]
            pose_coords = []
            pose_scores = []
            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size,
                                                               norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints': preds_img[k],
                        'kp_score': preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx': ids[k],
                        'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                    }
                )

            result = {
                'imgname': im_name,
                'result': _result
            }

            # final_result.append(result)
            img = vis_frame(orig_img, result, self.args)
            return img

    def hmHeatMapToCoordinations(self, hm, cropped_boxes):
        eval_joints = [*range(0, 136)]
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        pose_coords = []
        pose_scores = []
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = self.heatmap_to_coord(hm[i][eval_joints], bbox, hm_shape=hm_size,
                                                           norm_type=norm_type)
            pose_coords.append(pose_coord)
            pose_scores.append(pose_score)
        return pose_coords, pose_scores

    def doOneFrameFull(self,bundle):
        boxes, scores, ids, hm, cropped_boxes, orig_img, im_name = self.doOneFrameHeat(bundle)
        if boxes is not None and len(boxes)*2 > self.maxHands:
            if self.maxHands != len(boxes)*2:
                try:
                    self.hands.close()
                    del self.hands
                except Exception:
                    pass
                self.maxHands = len(boxes)*2
                self.hands = mp_hands.Hands(
                    max_num_hands=self.maxHands,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2)
        results = self.hands.process(orig_img)
        hands_out = []
        if results is not None and results.multi_hand_landmarks is not None and results.multi_handedness is not None:
            for i in range(len(results.multi_hand_landmarks)):
                m1 = results.multi_handedness[i]
                m2 = results.multi_hand_landmarks[i]
                hands_out.append((m1,m2))

        pose_coords = None
        pose_scores = None
        if boxes is not None:
            pose_coords, pose_scores = self.hmHeatMapToCoordinations(hm, cropped_boxes)
            pose_scores = np.asarray(pose_scores)
            pose_coords = np.asarray(pose_coords)
        if ids is not None:
            if args.pose_track:
                ids = np.asarray(ids)
            else:
                id_temp = []
                for i in range(pose_coords.shape[0]):
                    id_temp.append(0)
                ids = np.asarray(id_temp).reshape((-1,))
        if boxes is not None:
            if torch.is_tensor(boxes):
                boxes = self.toNumpy(boxes)
            else:
                boxes = np.asarray(boxes)
        if cropped_boxes is not None:
            if torch.is_tensor(cropped_boxes):
                cropped_boxes = self.toNumpy(cropped_boxes)
            else:
                cropped_temp = []
                for c in cropped_boxes:
                    cropped_temp.append(self.toNumpy(c))
                cropped_boxes = np.asarray(cropped_temp)
        if hm is not None:
            if torch.is_tensor(hm):
                hm = self.toNumpy(hm)
            else:
                hm = np.asarray(hm)
        if scores is not None:
            if torch.is_tensor(scores):
                scores = self.toNumpy(scores)
            else:
                scores_temp = []
                for s in scores:
                    scores_temp.append(self.toNumpy(s))
                scores = np.asarray(scores_temp)

        return boxes, scores, ids, hm, cropped_boxes, orig_img, im_name, pose_coords, pose_scores, hands_out

    def toNumpy(self, tensor):
        if tensor is not None:
            return tensor.detach().cpu().numpy()
        else:
            return None


def halpeToCocoSkeleton(pose_coords):
    # also usable for H36m skeleton!
    ret = np.zeros((17, pose_coords.shape[1]))
    ret[0, :] = pose_coords[19, :]  # Hip -> Hips
    ret[1, :] = pose_coords[12, :]  # RHip -> RighUpLeg
    ret[2, :] = pose_coords[14, :]  # RKnee -> RightLeg
    ret[3, :] = pose_coords[16, :]  # RAnkle -> RightFoot
    ret[4, :] = pose_coords[11, :]  # LHip -> LeftUpLeg
    ret[5, :] = pose_coords[13, :]  # LKnee -> LeftLeg
    ret[6, :] = pose_coords[15, :]  # LAnkle -> LeftFoot
    ret[7, :] = (pose_coords[19, :] + pose_coords[18, :]) / 2  # (Hip+Neck) / 2 -> Spine
    ret[8, :] = pose_coords[18, :]  # Neck -> Spine1 (actually neck)
    ret[9, :] = pose_coords[0, :]  # Nose -> Neck1 (actually nose)
    ret[10, :] = pose_coords[17, :]  # Head -> HeadEndSite
    ret[11, :] = pose_coords[5, :]  # LShoulder -> LeftArm
    ret[12, :] = pose_coords[7, :]  # LElbow -> LeftForeArm
    ret[13, :] = pose_coords[9, :]  # LWrist -> LeftHand
    ret[14, :] = pose_coords[6, :]  # -> RightArm
    ret[15, :] = pose_coords[8, :]  # -> RightForeArm
    ret[16, :] = pose_coords[10, :]  # -> RightHand
    return ret

def halpeToOpenPoseSkeleton(pose_coords):
    ret = np.zeros((25, pose_coords.shape[1]))
    ret[0, :] = pose_coords[0, :]
    ret[1, :] = pose_coords[18, :]
    ret[2, :] = pose_coords[6, :]
    ret[3, :] = pose_coords[8, :]
    ret[4, :] = pose_coords[4, :]
    ret[5, :] = pose_coords[5, :]
    ret[6, :] = pose_coords[7, :]
    ret[7, :] = pose_coords[9, :]
    ret[8, :] = pose_coords[19, :]
    ret[9, :] = pose_coords[12, :]
    ret[10, :] = pose_coords[14, :]
    ret[11, :] = pose_coords[16, :]
    ret[12, :] = pose_coords[11, :]
    ret[13, :] = pose_coords[13, :]
    ret[14, :] = pose_coords[15, :]
    ret[15, :] = pose_coords[2, :]
    ret[16, :] = pose_coords[1, :]
    ret[17, :] = pose_coords[4, :]
    ret[18, :] = pose_coords[3, :]
    ret[19, :] = pose_coords[20, :]
    ret[20, :] = pose_coords[22, :]
    ret[21, :] = pose_coords[24, :]
    ret[22, :] = pose_coords[21, :]
    ret[23, :] = pose_coords[23, :]
    ret[24, :] = pose_coords[25, :]
    return ret


halpe_keypoints = {0: "Nose",
                   1: "LEye",
                   2: "REye",
                   3: "LEar",
                   4: "REar",
                   5: "LShoulder",
                   6: "RShoulder",
                   7: "LElbow",
                   8: "RElbow",
                   9: "LWrist",
                   10: "RWrist",
                   11: "LHip",
                   12: "RHip",
                   13: "LKnee",
                   14: "Rknee",
                   15: "LAnkle",
                   16: "RAnkle",
                   17: "Head",
                   18: "Neck",
                   19: "Hip",
                   20: "LBigToe",
                   21: "RBigToe",
                   22: "LSmallToe",
                   23: "RSmallToe",
                   24: "LHeel",
                   25: "RHeel", }


# //face
# {26-93, 68 Face Keypoints}
# //left hand
# {94-114, 21 Left Hand Keypoints}
# //right hand
# {115-135, 21 Right Hand Keypoints}

class RealtimeSequence:

    def __init__(self, historyFrames=140, calculateTrajectory=True, alphaPoseHelper=None):
        self.historyFrames = historyFrames
        self.persondict = {}
        self.ie = 0
        self.calculateTrajectory = calculateTrajectory
        self.aH = alphaPoseHelper if alphaPoseHelper is not None else AlphaPoseHelper(justLoad=True)



    def newFrame(self, frame_input, store_image=True):
        boxes, scores, ids, hm, cropped_boxes, orig_img, im_name, pose_coords, pose_scores, hands = frame_input
        #print(ids)
        if ids is None:
            ids = []
        ret = {}
        hands = self.repackHands(hands=hands, orig_img=orig_img)
        self.image_dimensions = orig_img.shape
        self.org_g = orig_img.copy()
        for i, id in enumerate(ids):


            if id in self.persondict:
                person = self.persondict[id]
            else:
                person = {"id": id, "seen": [], "last_seen": int(self.ie), "frames": [], "pose_coords": [],
                          "pose_scores": [], "predicted_3d_pos": [], "predicted_3d_traj": [],"3d_pose_coords":[],
                          "3d_hand_z_left":[], "3d_hand_z_right":[]}
            if self.ie - person["last_seen"] > 1:
                person = self.interpolatePosition(person, pose_coords[i], pose_scores[i], self.ie - person["last_seen"])
                person = self.interpolate3DPosition(person, halpeToCocoSkeleton(pose_coords[i]), pose_scores[i], self.ie - person["last_seen"])
            person["pose_coords"].append(pose_coords[i])
            person["3d_pose_coords"].append(halpeToCocoSkeleton(pose_coords[i]))
            person["pose_scores"].append(pose_scores[i])
            person = self.calculateTraj(person, orig_img)
            person = self.calcHands(person,hands,orig_img)

            person["last_seen"] = self.ie
            person["seen"].append(int(self.ie))

            if store_image:
                person["frames"].append(orig_img)
            person = self.trimPerson(person)
            self.persondict[id] = person
            ret[id] = self.createReturnDict(person, store_image)
            #cv2.imshow("img", orig_img)
            #cv2.waitKey(1)
        self.ie += 1
        return ret

    def drawHands(self,hands,orig_img):
        for score,points in hands:
            for p in points:
                pass
                #cv2.circle(orig_img, (int(p[0]), int(p[1])), radius=2, color=(0, 0, 255),
                #           thickness=-1)

    def repackHands(self, hands, orig_img):
        out = []

        for hand in hands:
            ha = np.zeros((21,3))
            e1 = hand[1].landmark
            for i in range(len(e1)):
                ha[i, 0] = e1[i].x
                ha[i, 1] = e1[i].y
                ha[i, 2] = e1[i].z
            ha2 = ha[:,:2] * [orig_img.shape[1], orig_img.shape[0]]
            #ha2 = denormalize_screen_coordinates(ha[:,:2],orig_img.shape[1],orig_img.shape[0])
            ha = np.concatenate((ha2,ha[:,2].reshape((21,1))),axis=1)
            out.append((hand[0].classification[0].score,ha))

        return out

    def calcHands(self,person,hands,orig_img):
        pos = person["pose_coords"][-1]
        right_hand = pos[115:136] / [orig_img.shape[1], orig_img.shape[0]]
        left_hand = pos[94:115] / [orig_img.shape[1], orig_img.shape[0]]

        best_score_r, best_hand_r = self.calc_one_hand(right_hand,hands, orig_img)
        best_score_l, best_hand_l = self.calc_one_hand(left_hand, hands, orig_img)
        if best_score_l < 0.1:
            points = best_hand_l[1]
            person["3d_hand_z_left"].append(points)
        if best_score_r < 0.1:
            points = best_hand_r[1]
            person["3d_hand_z_right"].append(points)


        return person

    def calc_one_hand(self,check_hand,hands,orig_img):
        best_hand = None
        best_score = 1000000
        for score, points in hands:
            test_points = points[:, :2].copy() / [orig_img.shape[1], orig_img.shape[0]]
            distance = np.linalg.norm(check_hand - test_points)
            if distance < best_score:
                best_hand = (score, points)
                best_score = distance
        return best_score,best_hand


    def createReturnDict(self, person, withImage=True):
        ret = {"id": person["id"],"image_dimensions":self.image_dimensions,
               "current_2d_pose": person["pose_coords"][-1] / [self.image_dimensions[1],self.image_dimensions[0]],
               "current_2d_score": person["pose_scores"][-1]}
        if withImage:
            ret["current_frame"] = person["frames"][-1]
        if self.calculateTrajectory:
            ret["current_3d_pose"] = person["predicted_3d_pos"][-1] / [self.image_dimensions[1],self.image_dimensions[0], 1]

            hand_l_z = person["3d_hand_z_left" ][-1][:,2].reshape((21,1)) * 2 if len(person["3d_hand_z_left" ]) > 0 else np.zeros((21, 1))
            hand_r_z = person["3d_hand_z_right"][-1][:,2].reshape((21,1)) * 2 if len(person["3d_hand_z_right"]) > 0 else np.zeros((21, 1))

            hand_l_z = hand_l_z + ret["current_3d_pose"][13, 2]
            hand_r_z = hand_r_z + ret["current_3d_pose"][16, 2]

            pos = person["pose_coords"][-1]
            right_hand = np.concatenate((pos[115:136],hand_r_z),axis=1) / [self.image_dimensions[1],self.image_dimensions[0], 1]
            left_hand = np.concatenate((pos[94:115],hand_l_z),axis=1) / [self.image_dimensions[1],self.image_dimensions[0], 1]


            ret["current_3d_pose"] = np.concatenate((ret["current_3d_pose"],left_hand,right_hand),axis=0)

            ret["current_3d_traj"] = person["predicted_3d_traj"][-1]
        return ret

    def interpolate_points(self, p1, p2, n_steps=3):
        """Helper function that calculates the interpolation between two points"""
        # interpolate ratios between the points.txt
        ratios = np.linspace(p1, p2, num=n_steps + 2)
        ratios = ratios[1:]
        ratios = ratios[:-1]
        # linear interpolate vectors
        # vectors = (1.0 - ratios) * p1 + ratios * p2
        return ratios

    def interpolatePosition(self, person, new_coords, new_scores, num):
        last_coords = person["pose_coords"][-1]
        last_scores = person["pose_scores"][-1]
        inter_coords = self.interpolate_points(last_coords, new_coords, num)
        inter_scores = self.interpolate_points(last_scores, new_scores, num)
        #print(inter_coords.shape, inter_scores.shape)
        for i in range(inter_coords.shape[0]):
            person["pose_coords"].append(inter_coords[i])
            person["pose_scores"].append(inter_scores[i])
        return person

    def interpolate3DPosition(self, person, new_coords, new_scores, num):
        last_coords = person["3d_pose_coords"][-1]
        last_scores = person["pose_scores"][-1]
        inter_coords = self.interpolate_points(last_coords, new_coords, num)
        inter_scores = self.interpolate_points(last_scores, new_scores, num)
        #print(inter_coords.shape, inter_scores.shape)
        for i in range(inter_coords.shape[0]):
            person["3d_pose_coords"].append(inter_coords[i])
            person["pose_scores"].append(inter_scores[i])
        return person

    def interpolateTrajectory(self, person, newPos, newTraj, num):
        last_pos = person["predicted_3d_pos"][-1]
        last_traj = person["predicted_3d_traj"][-1]
        inter_pos = self.interpolate_points(last_pos, newPos, num)
        inter_traj = self.interpolate_points(last_traj, newTraj, num)
        for i in range(inter_pos.shape[0]):
            person["predicted_3d_pos"].append(inter_pos[i])
            person["predicted_3d_traj"].append(inter_traj[i])
        return person

    def calculateTraj(self, person, orig_img):
        predicted_3d_pos, predicted_3d_traj = self.aH.do2Dto3D(np.asarray(person["3d_pose_coords"]), orig_img.shape[1],orig_img.shape[0])
        predicted_3d_pos = predicted_3d_pos[-1]
        #predicted_3d_pos2 = denormalize_screen_coordinates(predicted_3d_pos[:,:2],orig_img.shape[1],orig_img.shape[0])


        predicted_3d_pos = np.concatenate((person["3d_pose_coords"][-1],predicted_3d_pos[:,2].reshape((17,1))),axis=1)

        predicted_3d_traj = predicted_3d_traj[-1]
        if self.ie - person["last_seen"] > 1:
            person = self.interpolateTrajectory(person, predicted_3d_pos, predicted_3d_traj,
                                                self.ie - person["last_seen"] - 1)
        person["predicted_3d_pos"].append(predicted_3d_pos)
        person["predicted_3d_traj"].append(predicted_3d_traj)
        return person

    def trimPerson(self, person):
        person["pose_coords"] = person["pose_coords"][-self.historyFrames:]
        person["3d_pose_coords"] = person["3d_pose_coords"][-self.historyFrames:]
        person["pose_scores"] = person["pose_scores"][-self.historyFrames:]
        person["predicted_3d_pos"] = person["predicted_3d_pos"][-self.historyFrames:]
        person["predicted_3d_traj"] = person["predicted_3d_traj"][-self.historyFrames:]
        person["frames"] = person["frames"][-self.historyFrames:]
        person["seen"] = person["seen"][-self.historyFrames:]
        return person


def alphaposeToDetectron(pose_coords, pose_scores, boxes):
    key = np.zeros((1, 4, 17))
    box = np.zeros((1, 5))
    for i in range(17):
        key[0, 0, i] = pose_coords[0, i, 0]
        key[0, 1, i] = pose_coords[0, i, 1]
        key[0, 3, i] = pose_scores[0, i, 0]
    for i in range(4):
        box[0, i] = boxes[0][i]
    box[0, 4] = 1
    return key, box


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def denormalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    #return X / w * 2 - [1, h / w]
    return (X + [1, h / w]) / 2 * w

if __name__ == '__main__':
    e1 = np.random.randint(low=0,high=500,size=(20,2)).astype(float)
    e1[:,0] *= 1.5
    print(e1.min(),e1.max(),e1.mean(),e1.std())
    e2 = normalize_screen_coordinates(e1,750,500)
    print(e2.min(), e2.max(), e2.mean(), e2.std())
    e3 = denormalize_screen_coordinates(e2,750,500)
    print(e3.min(), e3.max(), e3.mean(), e3.std())
    print((e3-e1).sum())
