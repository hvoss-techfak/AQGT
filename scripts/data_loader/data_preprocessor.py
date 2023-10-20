""" create data samples """
import math
import multiprocessing
import os
# sys.path.append("/mnt/98072f92-dbd5-4613-b3b6-f857bcdcdadc/Owncloud/Projekte/Promotion/SEEG-main")
import pickle
import threading
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# os.chdir("../")
from time import sleep

import lmdb
import lz4.frame
import numpy as np
import pyarrow
from tqdm import tqdm

from scripts import utils
from scripts.data_loader.motion_preprocessor import MotionPreprocessor
from scripts.utils import data_utils
from scripts.utils.data_utils import resampled_get_anno


# from scripts.utils.SkeletonHelper.VisualizeHelper import VisualizeCvClass


def toOccurenceMap(anno, key):
    out = np.zeros((34,))
    smooth = False
    for i, l1 in enumerate(anno):
        if l1[key] is not None:
            smooth = True
            out[i] = 1
    if smooth:
        for i in range(1, 34):
            out[i] = round((out[i - 1] + out[i]) / 2, 4)
        for i in reversed(range(0, 33)):
            out[i] = round((out[i] + out[i + 1]) / 2, 4)
    return out


def toEntityMap(entity_names):
    entity_list = {"Hufeisen": 0, "Stockwerk": 1, "Treppe": 2, "Fenster": 3, "Kirche": 4, "Tür": 5, "Aussenobjekte": 6,
                   "Lampen": 7, "Turm": 8, "Schale": 9, "Straße": 10, "Rathaus": 11, "Platz": 12, "Gebäude": 13,
                   "Kunstobjekt": 14, "Dach": 15, "Brunnen": 16, "Hecke": 17}
    entity = np.zeros((34,))
    for i, f_entity in enumerate(entity_names):
        try:
            entity[i] = entity_list[f_entity] + 1 if f_entity is not None else 0
        except Exception:
            traceback.print_exc()
            entity[i] = 0
    return entity


def toPhaseMap(phase_names):
    phase_list = {"prep": 0, "stroke": 1, "stroke ": 1,"stroke\n": 1, "retr": 2, "post.hold": 3, "pre.hold": 4, '': -1}
    ret = np.zeros((34,))
    for i, f_entity in enumerate(phase_names):
        try:
            ret[i] = phase_list[f_entity] + 1 if f_entity is not None else 0
        except Exception:
            traceback.print_exc()
            ret[i] = 0
    return ret


def toPhraseMap(phrase_names):
    phrase_list = {"beat": 0, "iconic": 1, "iconic ": 1, 'iconic \n': 1, 'iconic\n\n': 1, "deictic": 2, "deictic-beat": 2,
                   "deictic-discourse": 2,
                   "iconic-deictic": 3, "discourse": 4, "discourse/beat": 4, "discourse/indexing": 4,
                   "discourse-beat": 4, "discourse-iconic": 4, "move": 5, "iconic-beat": 5, "iconic-deictic-beat": 7,
                   'unclear': -1}
    ret = np.zeros((34,))
    for i, f_entity in enumerate(phrase_names):
        try:
            ret[i] = phrase_list[f_entity] + 1 if f_entity is not None else 0
        except Exception:
            traceback.print_exc()
            ret[i] = 0
    return ret


def toPositionMap(phrase_names):
    phrase_list = {'links': 0, 'drauf': 1, 'rechts': 2, 'davor': 3, 'drinnen': 4, 'hier': 5, 'daneben': 6,
                   'dahinter': 7, 'zusammen': 8, 'drunter': 9, 'hin': 10, 'drumherum': 11, 'zwischen': 12}
    ret = np.zeros((34,))
    for i, f_entity in enumerate(phrase_names):
        try:
            ret[i] = phrase_list[f_entity] + 1 if (f_entity is not None and f_entity in phrase_list.keys()) else 0
        except Exception:
            traceback.print_exc()
            ret[i] = 0
    return ret


def toShapeMap(phrase_names):
    phrase_list = {'B_spread': 0, 'B_loose_spread': 1, 'G': 2, '5_bent': 3, 'C': 4, 'B_spread_loose': 1, 'G_loose': 5,
                   '5': 6, 'C_loose': 7, 'G_bent': 8, 'B': 9, 'C_large': 10, 'B_loose': 11, 'C_small': 12,
                   '5_loose': 13, 'O': 16, 'C_large_loose': 17, 'H_loose': 18, 'D': 19}
    ret = np.zeros((34,))
    for i, f_entity in enumerate(phrase_names):
        try:
            ret[i] = phrase_list[f_entity] + 1 if (f_entity is not None and f_entity in phrase_list.keys()) else 0
        except Exception:
            traceback.print_exc()
            ret[i] = 0
    return ret


def toWristMap(phrase_names):
    phrase_list = {'D-CE': 0, 'D-EK': 1, 'D-KO': 2, 'D-O': 3, 'D-C': 4, '0': 5}
    ret = np.zeros((34,))
    for i, f_entity in enumerate(phrase_names):
        try:
            ret[i] = phrase_list[f_entity] + 1 if (f_entity is not None and f_entity in phrase_list.keys()) else 0
        except Exception:
            traceback.print_exc()
            ret[i] = 0
    return ret


def toExtendMap(phrase_names):
    phrase_list = {'0': 0, 'SMALL': 1, 'MEDIUM': 2, 'LARGE': 3}
    ret = np.zeros((34,))
    for i, f_entity in enumerate(phrase_names):
        try:
            ret[i] = phrase_list[f_entity] + 1 if (f_entity is not None and f_entity in phrase_list.keys()) else 0
        except Exception:
            traceback.print_exc()
            ret[i] = 0
    return ret


def toPracticeMap(phrase_names):
    phrase_list = {'shaping': 1, 'indexing': 2, 'shaping-modelling': 3, 'grasping-indexing': 4, 'drawing': 5,
                   'modelling': 6, '0': 0, 'hedging': 7, 'grasping': 8, 'sizing': 9, 'counting': 10, 'action': 11,
                   'drawing-modelling': 3, 'modelling-drawing': 3, 'modelling-indexing': 12, 'shaping-sizing': 13}
    ret = np.zeros((34,))
    for i, f_entity in enumerate(phrase_names):
        try:
            ret[i] = phrase_list[f_entity] + 1 if (f_entity is not None and f_entity in phrase_list.keys()) else 0
        except Exception:
            traceback.print_exc()
            ret[i] = 0
    return ret


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride,
                 pose_resampling_fps=25, mean_pose=np.zeros((48, 3)), mean_dir_vec=np.zeros((48, 3)),
                 disable_filtering=False):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_pose = mean_pose
        self.mean_dir_vec = mean_dir_vec
        self.disable_filtering = True
        self.lmdb_dir = clip_lmdb_dir
        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.spectrogram_sample_length = data_utils.calc_spectrogram_length_from_motion_length(self.n_poses,
                                                                                               self.skeleton_resampling_fps)
        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 850  # in MB
        map_size <<= 20  # in B
        self.map_size = map_size
        self.out_lmdb_dir = out_lmdb_dir
        os.makedirs(out_lmdb_dir, exist_ok=True)
        self.dst_lmdb_env = lmdb.open(self.out_lmdb_dir, map_size=self.map_size, lock=True)
        self.n_out_samples = 0
        self.lock = threading.Lock()
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()



    def writeTread(self):
        try:
            txn = self.dst_lmdb_env.begin(write=True)
            ie = 0
            while True:
                try:
                    k, v = self.queue.get()
                    if k == -1 and v == -1:
                        print("shutting down")
                        break
                    if k is not None and v is not None:
                        txn.put(k, v)
                    ie += 1
                    if ie % 1000 == 0:
                        txn.commit()
                        del txn
                        txn = self.dst_lmdb_env.begin(write=True)
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()

    def run(self):

        n_filtered_out = defaultdict(int)
        src_txn = self.src_lmdb_env.begin(write=False)

        tlist = []
        print("processing with",os.cpu_count(),"threads")
        exec = ProcessPoolExecutor(max_workers=os.cpu_count())
        exec2 = ThreadPoolExecutor(max_workers=1)
        exec2.submit(self.writeTread)
        tbar = tqdm(total=self.n_videos)

        def process_list(force=False):
            for t in list(tlist):
                if t.done() or force:
                    try:
                        filtered_result,out = t.result()
                        if out is not None:
                            for sample_words_list, sample_skeletons_list, sample_audio_list, aux_info, sample_anno_list in out:
                                if len(sample_skeletons_list) > 0:
                                    self.putData(sample_words_list, sample_skeletons_list, sample_audio_list, aux_info, sample_anno_list)
                        tlist.remove(t)
                        tbar.update(1)
                        for type in filtered_result.keys():
                            n_filtered_out[type] += filtered_result[type]
                    except Exception:
                        traceback.print_exc()

        cursor = src_txn.cursor()

        with self.src_lmdb_env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))

        for key in keys:

            tlist.append(exec.submit(self.doclip, key, self.lmdb_dir,self.skeleton_resampling_fps,self.n_poses,self.subdivision_stride,self.audio_sample_length,self.mean_pose))
            while len(tlist) >= 10:
                process_list()
                sleep(1)
                tbar.set_description(str(self.n_out_samples) + " : " + str(self.queue.qsize()))

        process_list(force=True)
        exec.shutdown()
        self.queue.put((-1, -1))

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                print('{}: {}'.format(type, n_filtered))
                n_total_filtered += n_filtered
            print('no. of excluded samples: {} ({:.1f}%)'.format(
                n_total_filtered, 100 * n_total_filtered / max(1, (txn.stat()['entries'] + n_total_filtered))))

        # close db
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

    @staticmethod
    def doclip(key, lmdb_dir,skeleton_resampling_fps,n_poses,subdivision_stride,audio_sample_length,mean_pose):
        src_lmdb_env = None
        try:
            src_lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
            src_txn = src_lmdb_env.begin(write=False)
            value = src_txn.cursor().get(key)

            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            n_filtered_out = defaultdict(int)
            out = []
            for clip_idx, clip in enumerate(clips):
                filtered_result,sample_words_list, sample_skeletons_list, sample_audio_list, aux_info, sample_anno_list = DataPreprocessor._sample_from_clip(vid, clip,skeleton_resampling_fps,n_poses,subdivision_stride,audio_sample_length,mean_pose)
                out.append((sample_words_list, sample_skeletons_list, sample_audio_list, aux_info, sample_anno_list))
                for type in filtered_result.keys():
                    n_filtered_out[type] += filtered_result[type]
            return n_filtered_out,out
        except Exception:
            return {},None
        finally:
            if src_lmdb_env is not None:
                try:
                    src_lmdb_env.close()
                except Exception:
                    traceback.print_exc()

    @staticmethod
    def normalize_dir_vec(dir_vec, mean_dir_vec):
        return dir_vec - mean_dir_vec

    @staticmethod
    def get_words_in_time_range(word_list, start_time, end_time, fps=25):
        words = []

        start_time = (start_time / 15) * fps
        end_time = (end_time / 15) * fps
        for word in word_list:
            _, word_s, word_e = word[0], word[1], word[2]

            if word_s >= end_time:
                break

            if word_e <= start_time:
                continue

            words.append(word)

        return words

    @staticmethod
    def unnormalize_data(normalized_data, data_mean, data_std, dimensions_to_ignore):
        """
        this method is from https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
        """
        T = normalized_data.shape[0]
        D = data_mean.shape[0]

        origData = np.zeros((T, D), dtype=np.float32)
        dimensions_to_use = []
        for i in range(D):
            if i in dimensions_to_ignore:
                continue
            dimensions_to_use.append(i)
        dimensions_to_use = np.array(dimensions_to_use)

        origData[:, dimensions_to_use] = normalized_data

        # potentially inefficient, but only done once per experiment
        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        origData = np.multiply(origData, stdMat) + meanMat

        return origData

    @staticmethod
    def _sample_from_clip(vid, clip,skeleton_resampling_fps,n_poses,subdivision_stride,audio_sample_length,mean_pose):
        clip_skeleton = clip['skeletons_3d']  # clip_data (48,3)
        clip_audio = clip['audio_feat']  # spectogram - currently unused
        clip_audio = clip_audio.reshape((clip_audio.shape[0], -1))
        clip_audio_raw = np.asarray(clip['audio_raw']).flatten()  # clip_audio at 16k hz (
        clip_word_list = clip['words']
        clip_s_f, clip_e_f = clip['start_frame_no'], clip['end_frame_no']
        clip_s_t, clip_e_t = clip['start_time'], clip['end_time']

        fps = round((clip_e_f - clip_s_f) / max(0.1, clip_e_t - clip_s_t), 4)
        clip_anno = clip["saga_annotation"] if "saga_annotation" in clip.keys() else None
        n_filtered_out = defaultdict(int)

        # skeleton resampling
        clip_skeleton = utils.data_utils.resample_pose_seq(clip_skeleton, clip_e_t - clip_s_t,
                                                           skeleton_resampling_fps)

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_words_list = []
        sample_audio_list = []
        sample_anno_list = []

        num_subdivision = math.floor(
            (len(clip_skeleton) - n_poses)
            / subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * subdivision_stride
            fin_idx = start_idx + n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            subdivision_start_time = clip_s_f + start_idx  # / self.skeleton_resampling_fps
            subdivision_end_time = clip_s_f + fin_idx  # / self.skeleton_resampling_fps
            sample_words = DataPreprocessor.get_words_in_time_range(word_list=clip_word_list,
                                                        start_time=subdivision_start_time,
                                                        end_time=subdivision_end_time, fps=fps)

            if clip_anno is not None:
                sample_anno = resampled_get_anno(clip_anno, start_idx, fin_idx, current_fps=15, original_fps=25)
                entity_map = toEntityMap([f["R_S_Semantic_Entity"] for f in sample_anno])
                occurence_map = toOccurenceMap(sample_anno, "R_S_Semantic_Entity")
                l_phase = toPhaseMap([f["R_G_Left_Phase"] for f in sample_anno])
                r_phase = toPhaseMap([f["R_G_Right_Phase"] for f in sample_anno])

                l_phrase = toPhraseMap([f["R_G_Left_Phrase"] for f in sample_anno])
                r_phrase = toPhraseMap([f["R_G_Right_Phrase"] for f in sample_anno])

                l_position = toPositionMap([f["R_G_Left_Semantic_Position"] for f in sample_anno])
                r_position = toPositionMap([f["R_G_Right_Semantic_Position"] for f in sample_anno])
                s_position = toPositionMap([f["R_S_Speech_Semantic_Position"] for f in sample_anno])

                l_hand_shape = toShapeMap([f["LeftHandShape"] for f in sample_anno])
                r_hand_shape = toShapeMap([f["RightHandShape"] for f in sample_anno])

                l_wrist_distance = toWristMap([f["LeftWristDistance"] for f in sample_anno])
                r_wrist_distance = toWristMap([f["RightWristDistance"] for f in sample_anno])

                l_extend = toExtendMap([f["LeftExtent"] for f in sample_anno])
                r_extend = toExtendMap([f["RightExtent"] for f in sample_anno])

                l_practice = toPracticeMap([f["LeftPractice"] for f in sample_anno])
                r_practice = toPracticeMap([f["RightPractice"] for f in sample_anno])

                sample_anno_list.append((entity_map, occurence_map, l_phase, r_phase, l_phrase, r_phrase, l_position,
                                         r_position, s_position, l_hand_shape, r_hand_shape, l_wrist_distance,
                                         r_wrist_distance, l_extend, r_extend, l_practice, r_practice))
            else:
                sample_anno_list.append((np.zeros(34, ) - 1,) * 17)

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + audio_sample_length
            if audio_end > clip_audio_raw.shape[0]:  # correct size mismatch between poses and audio


                sample_audio = clip_audio_raw[audio_start:clip_audio_raw.shape[0]]
                padded_data = np.zeros((audio_end - audio_start))
                padded_data[:sample_audio.shape[0]] = sample_audio

                sample_audio = padded_data
            else:
                sample_audio = clip_audio_raw[audio_start:audio_end]

            if len(sample_words) >= 0:

                sample_skeletons, filtering_message = MotionPreprocessor(sample_skeletons, mean_pose).get()
                is_correct_motion = (sample_skeletons != [])
                motion_info = {'vid': vid,
                               'start_frame_no': clip_s_f + start_idx,
                               'end_frame_no': clip_s_f + fin_idx,
                               'start_time': subdivision_start_time,
                               'end_time': subdivision_end_time,
                               'is_correct_motion': is_correct_motion, 'filtering_message': filtering_message}

                if is_correct_motion:
                    sample_skeletons_list.append(sample_skeletons)
                    sample_words_list.append(sample_words)
                    sample_audio_list.append(sample_audio)
                    aux_info.append(motion_info)
                else:
                    n_filtered_out[filtering_message] += 1




        return n_filtered_out,sample_words_list, sample_skeletons_list, sample_audio_list, aux_info, sample_anno_list

    def putData(self, sample_words_list, sample_skeletons_list, sample_audio_list, aux_info, sample_anno_list):
        out = []
        for i, (words, poses, audio, aux, anno) in enumerate(zip(sample_words_list, sample_skeletons_list,
                                                                 sample_audio_list,
                                                                 aux_info, sample_anno_list)):
            try:
                # preprocessing for poses
                poses = np.asarray(poses)
                if poses.shape[0] > 0:
                    dir_vec = utils.data_utils.convert_pose_seq_to_dir_vec(poses.copy())
                    # print(self.mean_dir_vec.min(), self.mean_dir_vec.max(), np.std(self.mean_dir_vec), np.mean(self.mean_dir_vec), np.median(self.mean_dir_vec))
                    normalized_dir_vec = dir_vec - self.mean_dir_vec  # self.normalize_dir_vec(dir_vec, self.mean_dir_vec)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [words, poses, normalized_dir_vec, audio, np.zeros(1, ), aux, anno]

                    v = lz4.frame.compress(pyarrow.serialize(v).to_buffer())
                    while self.queue.qsize() > 1000:
                        print("waiting due to queue size")
                        sleep(1)
                    self.queue.put((k, v))

                    self.n_out_samples += 1
            except Exception:
                traceback.print_exc()

        # print(len(out))
        # for k,v in out:


def meanThread(key,lmdb_dir, ):
    try:
        src_lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        src_txn = src_lmdb_env.begin(write=False)
        value = src_txn.cursor().get(key)

        video = pyarrow.deserialize(value)
        vid = video['vid']
        clips = video['clips']
        for clip in clips:
            clip_skeleton = clip['skeletons_3d']  # clip_data (48,3)

            poses = np.asarray(clip_skeleton)

            dir_vec = utils.data_utils.convert_pose_seq_to_dir_vec(poses)
            return dir_vec.mean(axis=0), poses.mean(axis=0)
    except Exception:
        traceback.print_exc()
        return None,None


def getMeans(lmdb_dir):
    src_lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    src_txn = src_lmdb_env.begin(write=False)

    dirvec_list = []
    mean_list = []

    tlist = []
    exec = ProcessPoolExecutor(max_workers=24)

    with src_lmdb_env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    print(keys)
    src_lmdb_env.close()

    ie = 0
    for key in tqdm(keys):
        # video = pyarrow.deserialize(value)
        tlist.append(exec.submit(meanThread, key, lmdb_dir))
        #meanThread(value)

        ie += 1
        if ie % 1000 == 0:
            temp = []
            docheck = False
            if len(dirvec_list) > 0:
                dirtee = np.asarray(dirvec_list).mean(axis=0)
                docheck = True
            for t in tqdm(tlist):
                dir_vec, poses = t.result()
                if dir_vec is not None:
                    dirvec_list.append(dir_vec)
                    temp.append(dir_vec)
                    mean_list.append(poses)
            if docheck:
                temp = np.asarray(temp).mean(axis=0)
                print("difference to last_dirvec:", np.mean(np.abs(dirtee - temp)))
            tlist = []

    for t in tqdm(tlist):
        dir_vec, poses = t.result()
        dirvec_list.append(dir_vec)
        mean_list.append(poses)

    mean_list = np.asarray(mean_list).mean(axis=0)
    dirvec_list = np.asarray(dirvec_list).mean(axis=0)

    print(mean_list.shape)
    print(dirvec_list.shape)
    pickle.dump((mean_list, dirvec_list), open("means.p", "wb"))
    return mean_list, dirvec_list


if __name__ == '__main__':
    mean_pose, mean_dir_vec = getMeans("dataset/AQGT/dataset_train")
    print(mean_pose)
    print("--------")
    print(mean_dir_vec)
