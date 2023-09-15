import re

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d

from scripts.utils.SkeletonHelper.helper_function import deboneUpper, reboneUpper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'orange'), (1, 5, 'darkgreen'),
                       (5, 6, 'limegreen'), (6, 7, 'darkseagreen')]
dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length

ubs = 6
skeleton_parents = np.asarray(
    [-1, 0, 7 - ubs, 8 - ubs, 9 - ubs, 8 - ubs, 11 - ubs, 12 - ubs, 8 - ubs, 14 - ubs, 15 - ubs])
hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])
hand_parents_l = hand_parents_l + 17 - ubs
hand_parents_r = hand_parents_r + 17 + 21 - ubs
skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)


def normalize_string(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def remove_tags_marks(text):
    reg_expr = re.compile('<.*?>|[.,:;!?]+')
    clean_text = re.sub(reg_expr, '', text)
    return clean_text


def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec


def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = ((n_frames / fps) * 16000 - 1024) / 512 + 1
    return int(round(ret))


def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    expected_n = duration_in_sec * fps
    if int(expected_n) != len(poses):
        f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')

        x_new = np.arange(0, n, n / expected_n)
        interpolated_y = f(x_new)
        if hasattr(poses, 'dtype'):
            interpolated_y = interpolated_y.astype(poses.dtype)
        return interpolated_y
    return poses


def resampled_get_anno(anno, start_idx, end_idx, current_fps=15, original_fps=25):
    ret = []
    for i in range(start_idx, end_idx):
        ie = round((i / current_fps) * original_fps)
        ret.append(anno[ie if ie < len(anno) else len(anno) - 1])
        if ie >= len(anno):
            print("outside of range for value:", ie, "given", i)
    return ret


def time_stretch_for_words(words, start_time, speech_speed_rate):
    for i in range(len(words)):
        if words[i][1] > start_time:
            words[i][1] = start_time + (words[i][1] - start_time) / speech_speed_rate
        words[i][2] = start_time + (words[i][2] - start_time) / speech_speed_rate

    return words


def make_audio_fixed_length(audio, expected_audio_length):
    n_padding = expected_audio_length - len(audio)
    if n_padding > 0:
        audio = np.pad(audio, (0, n_padding), mode='symmetric')
    else:
        audio = audio[0:expected_audio_length]
    return audio


def convert_dir_vec_to_pose(vec):
    org_shape = vec.shape

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))
    vec = vec.copy()

    if len(vec.shape) == 3:
        for i in range(vec.shape[0]):
            vec[i] = reboneUpper(vec[i])
        return vec.reshape(org_shape)

    elif len(vec.shape) == 4:
        for i in range(vec.shape[0]):
            for j in range(vec.shape[1]):
                vec[i, j] = reboneUpper(vec[i, j])
        return vec.reshape(org_shape)

    else:
        assert False


def convert_pose_seq_to_dir_vec(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape((-1, 53, 3))

    pose = pose.copy()
    if len(pose.shape) == 3:
        for i in range(pose.shape[0]):
            pose[i] = deboneUpper(pose[i])
        return pose
    else:
        print(pose)
        print(pose.shape)
        assert False
