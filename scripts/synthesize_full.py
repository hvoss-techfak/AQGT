import copy
import datetime
import logging
import math
import os
import pickle
import random
import sys
import time

import librosa
import librosa.display
import lmdb
import numpy as np
import pyarrow
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, BertTokenizer, BertModel
from transformers.utils import PaddingStrategy

import datetime
import itertools
import os.path
import pprint
import random



from collections import OrderedDict
from copy import copy
from random import shuffle



import cv2
import matplotlib
import torch.nn.functional as F
import torch.random
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from torchelie.optim import RAdamW


from lookahead import Lookahead
from model.extra_losses import V7_loss
from model.AQGT_plus import AqgtPlusGenerator, MyLoopDiscriminator
from utils.SkeletonHelper.VisualizeHelper import VisualizeCvClass
from train_eval.train_AQGT_plus import train_dis, train_gen
from utils.data_utils import convert_dir_vec_to_pose
from utils.vocab_utils import build_vocab

from config.parse_args import parse_args
from scripts.data_loader.data_preprocessor import toEntityMap, toOccurenceMap, toPhaseMap, toPhraseMap, \
    toPositionMap, toShapeMap, toWristMap, toExtendMap, toPracticeMap


import utils

from data_loader.data_preprocessor import toEntityMap, toOccurenceMap, toPhaseMap, toPhraseMap, \
    toPositionMap, toShapeMap, toWristMap, toExtendMap, toPracticeMap
from data_loader.lmdb_data_loader import proc_audio
from model.VQVAE_2_audio import VQ_VAE_2_audio
from scripts.train_lightning import load_lightning_model
from utils.data_utils import extract_melspectrogram, resampled_get_anno
from utils.train_utils import create_video_and_save, set_logger
from config.parse_args import parse_args


device = torch.device("cpu")
audio_vqvae = VQ_VAE_2_audio.load_from_checkpoint("pretrained/vqvae_audio/vqvae_audio.ckpt", strict=False).cuda().eval()

Wav2Vec2processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
Wav2Vec2model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")


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


def generate_gestures(args, pose_decoder, lang_model, audio, words, annotation_data=None, audio_sr=16000, vid=None,
                      seed_seq=None, fade_out=False, mean_dir_vec=None, embedding=None, fps=None, modifier=None):
    out_list = []
    anno_list = []

    n_frames = args.n_poses
    n_sampling_rate = args.motion_resampling_framerate

    clip_length = len(audio) / audio_sr

    Wav2Vec2model.eval()
    for p in Wav2Vec2model.parameters():
        p.requires_grad = False

    bert_model.eval()
    for p in bert_model.parameters():
        p.requires_grad = False

    use_spectrogram = False
    if args.model == 'speech2gesture':
        use_spectrogram = True

    # pre seq
    pre_seq = torch.zeros((1, n_frames, len(mean_dir_vec) + 1))
    if seed_seq is not None:
        pre_seq[0, 0:args.n_pre_poses, :-1] = torch.Tensor(seed_seq[0:args.n_pre_poses])
        pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for seed poses

    sr = 16000
    spectrogram = None
    if use_spectrogram:
        # audio to spectrogram
        spectrogram = extract_melspectrogram(audio, sr)

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    spectrogram_sample_length = int(round(unit_time * sr / 512))
    audio_sample_length = int(unit_time * audio_sr)
    end_padding_duration = 0

    # prepare speaker input
    if args.z_type == 'speaker':
        if not vid:
            vid = random.randrange(pose_decoder.z_obj.n_words)
        # print('vid:', vid)
        vid = torch.LongTensor([vid]).to(device)
    else:
        vid = None

    # print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

    out_dir_vec = None
    start = time.time()
    for i in tqdm(range(0, num_subdivision)):
        # for i in tqdm(range(0, 2)):
        start_time = i * stride_time
        end_time = start_time + unit_time

        start_frame = int(i * stride_time * 15)

        fin_idx = start_frame + n_frames

        # sample_skeletons = clip_skeleton[start_idx:fin_idx]
        # subdivision_start_time = clip_s_f + start_idx  # / self.skeleton_resampling_fps
        # subdivision_end_time = clip_s_f + fin_idx

        # prepare spectrogram input
        in_spec = None
        if use_spectrogram:
            # prepare spec input
            audio_start = math.floor(start_time / clip_length * spectrogram.shape[0])
            audio_end = audio_start + spectrogram_sample_length
            in_spec = spectrogram[:, audio_start:audio_end]
            in_spec = torch.from_numpy(in_spec).unsqueeze(0).to(device)

        # prepare audio input
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        if modifier[2] is None:
            in_audio *= 0

        if len(in_audio) < audio_sample_length:
            if i == num_subdivision - 1:
                end_padding_duration = audio_sample_length - len(in_audio)
            in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
        audio_numpy = in_audio.copy().astype(np.float32)
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()

        audio_beat = proc_audio(in_audio.squeeze(0), audio_numpy)

        time_seq = audio_beat.unsqueeze(0)

        # prepare text input
        word_seq = get_words_in_time_range(word_list=words, start_time=start_frame, end_time=fin_idx, fps=fps)
        sent = ' '.join([w[0] for w in word_seq])
        if len(sent) > 0:
            print(start_frame / 15, sent)

        # prepare pre seq
        if i > 0:
            pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
            pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq = pre_seq.float().to(device)
        pre_seq_partial = pre_seq[0, 0:args.n_pre_poses, :-1].unsqueeze(0)
        pose_decoder.eval()

        # synthesize
        # print(in_text_padded)

        with torch.no_grad():
            if args.model == 'multimodal_context':
                in_audio1 = in_audio.reshape(1, -1)
                in_audio1 -= in_audio1.min()
                in_audio1 /= max(0.0000001, in_audio1.max()) / 2
                in_audio1 -= 1
                # print(in_audio.min(),in_audio.max())
                in_audio1 = in_audio1[:, :int((in_audio1.shape[1] // 4000) * 4000)].reshape((in_audio1.shape[0], -1, 4000))

                with torch.no_grad():
                    entity_map, occurence_map, l_phase, r_phase, l_phrase, r_phrase = None, None, None, None, None, None
                    if annotation_data is not None:
                        # print(start_frame)
                        sample_anno = resampled_get_anno(annotation_data, start_frame, fin_idx, current_fps=args.motion_resampling_framerate, original_fps=25)
                        entity_map = torch.from_numpy(toEntityMap([f["R_S_Semantic_Entity"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        occurence_map = torch.from_numpy(toOccurenceMap(sample_anno, "R_S_Semantic_Entity")).to(in_audio.device).unsqueeze(0)
                        l_phase = torch.from_numpy(toPhaseMap([f["R_G_Left_Phase"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        r_phase = torch.from_numpy(toPhaseMap([f["R_G_Right_Phase"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)

                        l_phrase = torch.from_numpy(toPhraseMap([f["R_G_Left_Phrase"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        # if l_phrase.max() > 0:
                        #    print("at second:",(start_frame/15),"found:",l_phrase.max())
                        r_phrase = torch.from_numpy(toPhraseMap([f["R_G_Right_Phrase"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)

                        l_position = torch.from_numpy(toPositionMap([f["R_G_Left_Semantic_Position"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        r_position = torch.from_numpy(toPositionMap([f["R_G_Right_Semantic_Position"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        s_position = torch.from_numpy(toPositionMap([f["R_S_Speech_Semantic_Position"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        if l_position.max() > 0 or r_position.max() > 0 or s_position.max() > 0:
                            print(l_position.max(), r_position.max(), s_position.max())

                        l_hand_shape = torch.from_numpy(toShapeMap([f["LeftHandShape"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        r_hand_shape = torch.from_numpy(toShapeMap([f["RightHandShape"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)

                        l_wrist_distance = torch.from_numpy(toWristMap([f["LeftWristDistance"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        r_wrist_distance = torch.from_numpy(toWristMap([f["RightWristDistance"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)

                        l_extend = torch.from_numpy(toExtendMap([f["LeftExtent"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        r_extend = torch.from_numpy(toExtendMap([f["RightExtent"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)

                        l_practice = torch.from_numpy(toPracticeMap([f["LeftPractice"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        r_practice = torch.from_numpy(toPracticeMap([f["RightPractice"] for f in sample_anno])).to(in_audio.device).unsqueeze(0)
                        anno = torch.cat((entity_map, occurence_map, l_phase, r_phase, l_phrase, r_phrase, l_position, r_position, s_position, l_hand_shape, r_hand_shape, l_wrist_distance, r_wrist_distance, l_extend, r_extend, l_practice, r_practice), dim=0).unsqueeze(0)
                        for idx, src_values, dst_value in modifier[3]:
                            for id in idx:
                                if id == -1:
                                    anno[:, :, :] = dst_value
                                else:
                                    for sv in src_values:
                                        if sv is None:
                                            anno[:, id, :] = dst_value + 1
                                        else:
                                            anno[:, id, :][anno[:, id, :] == sv + 1] = dst_value + 1
                        anno = anno.float()
                    else:
                        anno = torch.zeros(1, 17, 34, device=in_audio.device).float() - 1
                        # print("adding annotation")
                        # print(anno.shape)
                    audio_var_bottom = torch.zeros((in_audio1.shape[0], in_audio1.shape[1], 16 * 16),
                                                   device=in_audio1.device)
                    audio_var_top = torch.zeros((in_audio1.shape[0], in_audio1.shape[1], 8 * 8), device=in_audio.device)
                    for i in range(in_audio1.shape[1]):
                        in_data = in_audio1[:, i, :]
                        in_temp = torch.zeros((in_data.shape[0], 4096), device=in_audio1.device)
                        in_temp[:, :4000] = in_data
                        in_temp = torch.moveaxis(in_temp.reshape((in_data.shape[0], 64, 64, 1)), 3, 1)

                        quant_t, quant_b, diff_a, _, _ = audio_vqvae.generator.encode(in_temp.cuda())
                        quant_b = quant_b.reshape(audio_var_bottom.shape[0], 16 * 16)
                        quant_t = quant_t.reshape(audio_var_top.shape[0], 8 * 8)
                        # print(quant_b.shape)
                        # print(quant_t.shape)

                        audio_var_bottom[:, i, :] = quant_b
                        audio_var_top[:, i, :] = quant_t

                    audio_var_bottom = audio_var_bottom.flatten().reshape(1, -1)
                    audio_var_top = audio_var_top.flatten().reshape(1, -1)

                ad1 = Wav2Vec2processor(in_audio.reshape(1, -1), sampling_rate=16000, return_tensors="pt")
                input_values = ad1.input_values.squeeze(0)

                log = Wav2Vec2model(input_values=input_values).logits

                encoded_input = tokenizer(sent, return_tensors='pt', padding=PaddingStrategy.LONGEST, max_length=500, truncation=True, )
                output = bert_model(**encoded_input)["pooler_output"]

                out_dir_vec, *_ = pose_decoder(pre_seq, time_seq, output, log, audio_var_bottom, audio_var_top, vid, anno)
            elif args.model == 'joint_embedding':
                _, _, _, _, _, _, out_dir_vec = pose_decoder(in_text_padded, in_audio, pre_seq_partial, None, 'speech')
            elif args.model == 'seq2seq':
                words_lengths = torch.LongTensor([in_text.shape[1]]).to(device)
                out_dir_vec = pose_decoder(in_text, words_lengths, pre_seq_partial, None)
            elif args.model == 'speech2gesture':
                out_dir_vec = pose_decoder(in_spec, pre_seq_partial)
            else:
                assert False

        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

            anno_list[-1] = anno_list[-1][:-args.n_pre_poses]
        anno_list.append(anno.squeeze(0).transpose(1, 0).cpu().numpy())
        out_list.append(out_seq)

    # print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    anno_out = np.vstack(anno_list)
    out_dir_vec = np.vstack(out_list)

    # additional interpolation for seq2seq
    if args.model == 'seq2seq':
        n_smooth = args.n_pre_poses
        for i in range(num_subdivision):
            start_frame = args.n_pre_poses + i * (args.n_poses - args.n_pre_poses) - n_smooth
            if start_frame < 0:
                start_frame = 0
                end_frame = start_frame + n_smooth * 2
            else:
                end_frame = start_frame + n_smooth * 3

            # spline interp
            y = out_dir_vec[start_frame:end_frame]
            x = np.array(range(0, y.shape[0]))
            w = np.ones(len(y))
            w[0] = 5
            w[-1] = 5

            coeffs = np.polyfit(x, y, 3)
            fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
            interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
            interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

            out_dir_vec[start_frame:end_frame] = interpolated_y

    # fade out to the mean pose
    if fade_out:
        n_smooth = args.n_pre_poses
        start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
        end_frame = start_frame + n_smooth * 2
        if len(out_dir_vec) < end_frame:
            out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode='constant')
        out_dir_vec[end_frame - n_smooth:] = np.zeros((len(args.mean_dir_vec)))  # fade out to mean poses

        # interpolation
        y = out_dir_vec[start_frame:end_frame]
        x = np.array(range(0, y.shape[0]))
        w = np.ones(len(y))
        w[0] = 5
        w[-1] = 5
        coeffs = np.polyfit(x, y, 2, w=w)
        fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
        interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
        interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

        out_dir_vec[start_frame:end_frame] = interpolated_y

    return out_dir_vec, anno_out


def main(args, mode, checkpoint_path, option, data_version=2):
    args, generator, loss_fn, lang_model, speaker_model, out_dim = load_lightning_model(args, args.checkpoint_path)
    generator = generator.eval()
    test_data_path = args.test_data_path[1]

    from torch import nn
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(lang_model.word_embedding_weights), freeze=True)

    # load mean vec
    mean_pose, mean_dir_vec = pickle.load(open("means.p", "rb"))
    mean_dir_vec = mean_dir_vec.flatten()
    save_path = 'output/generation_results'
    random.seed()

    # load clips and make gestures
    n_saved = 0
    save_counter = 0
    lmdb_env = lmdb.open(test_data_path, readonly=True, lock=False)
    handle = open('test_save_full.txt', 'a')
    sub_folder_save_name = 'test_full'
    if not os.path.exists(sub_folder_save_name):
        os.makedirs(sub_folder_save_name)
    ie = 0
    with lmdb_env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
        count_bar = tqdm(total=len(keys))
        for key in keys:
            count_bar.update(1)

            buf = txn.get(key)
            video = pyarrow.deserialize(buf)
            vid = video['vid']
            clips = video['clips']

            # select clip
            n_clips = len(clips)
            modifiers = []

            # name, text,audio,annotation_modifiers
            # always None = remove, integer = move timeline,
            # tuple = remove value by other value (only for annotations)
            # for this the tuple gets an index [0,16] and two values to replace, with remove value = None

            # don't forget to always add +1 to index, as zero is no gesture
            # iconic to deictic; 4,5; 1 -> 2
            # deictic to iconic; 4,5; 2 -> 1
            # always do iconic; 4,5; 0,1,2,3,4,5,6,7 -> 1
            # always do deictic 4,5; 0,1,2,3,4,5,6,7 -> 2

            # only strokes; 2,3; 0,2,3,4 -> 1
            # only prep; 2,3; 1,2,3,4 -> 0
            # always do strokes; 2,3; 0,1,2,3,4 -> 1

            # small to wide wrist gestures; 13,14; 1,2 -> 3
            # wide to small wrist gestures; 13,14; 3,2 -> 1

            # only wide wrist gestures; 13,14; 0,1,2 -> 3

            modifiers.append(["original", 0, 0, []])

            # modifiers.append(["no text", None, 0, []])
            # modifiers.append(["no audio", 0, None, []])

            #modifiers.append(["no anno", 0, 0, [((-1,), (-1,), -1), ]])

            # modifiers.append(["zero anno", 0, 0, [((-1,), (-1,), 0), ]])
            # modifiers.append(["one anno", 0, 0, [((-1,), (-1,), 1), ]])
            # modifiers.append(["two anno", 0, 0, [((-1,), (-1,), 2), ]])
            # modifiers.append(["three anno", 0, 0, [((-1,), (-1,), 3), ]])
            #
            # modifiers.append(["always do strokes", 0, 0, [((2, 3), (None,), 1), ]])
            # modifiers.append(["always do prep", 0, 0, [((2, 3), (None,), 0), ]])
            # modifiers.append(["always do beat", 0, 0, [((4, 5), (None,), 0), ]])
            # modifiers.append(["always do iconic", 0, 0, [((4, 5), (None,), 1), ]])
            # modifiers.append(["always do deictic", 0, 0, [((4, 5), (None,), 2), ]])
            #
            # modifiers.append(["always gesture right", 0, 0, [((6,7,8), (None,), 2), ]])
            # modifiers.append(["always gesture left", 0, 0, [((6, 7, 8), (None,), 0), ]])
            #
            # modifiers.append(["always gesture Kirche", 0, 0, [((0,), (None,), 4), ]])
            # modifiers.append(["always gesture Kunstobject", 0, 0, [((0,), (None,), 14), ]])
            # modifiers.append(["always gesture Fenster", 0, 0, [((0,), (None,), 3), ]])
            # modifiers.append(["always gesture Tür", 0, 0, [((0,), (None,), 5), ]])
            #
            # modifiers.append(["always iconic and stroke", 0, 0, [((4, 5), (None,), 1), ((2, 3), (None,), 1)]])
            # modifiers.append(["always iconic and prep", 0, 0, [((4, 5), (None,), 1), ((2, 3), (None,), 0)]])
            # modifiers.append(["always deictic and stroke", 0, 0, [((4, 5), (None,), 2), ((2, 3), (None,), 1)]])
            # modifiers.append(["always deictic and prep", 0, 0, [((4, 5), (None,), 2), ((2, 3), (None,), 0)]])

            # modifiers.append(["always iconic and stroke and Kirche", 0, 0, [((4, 5), (None,), 1), ((2, 3), (None,), 1),((0,), (None,), 4), ]])
            # modifiers.append(["always deictic and stroke and Kirche", 0, 0, [((4, 5), (None,), 2), ((2, 3), (None,), 1), ((0,), (None,), 4), ]])

            # modifiers.append(["always iconic and stroke and Kunstobject", 0, 0, [((4, 5), (None,), 1), ((2, 3), (None,), 1), ((0,), (None,), 14), ]])
            # modifiers.append(["always deictic and stroke and Kunstobject", 0, 0, [((4, 5), (None,), 2), ((2, 3), (None,), 1), ((0,), (None,), 14), ]])
            #
            # modifiers.append(["only deictic", 0, 0, [((4, 5), (1, 2, 3, 4, 5, 6, 7), 2), ]])
            # modifiers.append(["only iconic", 0, 0, [((4, 5), (1, 2, 3, 4, 5, 6, 7), 1), ]])
            # modifiers.append(["only strokes", 0, 0, [((2, 3), (0, 1, 2, 3, 4,), 1), ]])
            # modifiers.append(["only prep", 0, 0, [((2, 3), (0, 2, 3, 4,), 0), ]])
            # modifiers.append(["only beat", 0, 0, [((4, 5), (1, 2, 3, 4, 5, 6, 7), 0), ]])
            #
            # modifiers.append(["iconic and stroke", 0, 0, [((4, 5), (0, 1, 2, 3, 4, 5, 6, 7,), 1), ((2, 3), (0, 1, 2, 3, 4,), 1)]])
            # modifiers.append(["deictic and stroke", 0, 0, [((4, 5), (0, 1, 2, 3, 4, 5, 6, 7), 2), ((2, 3), (0, 1, 2, 3, 4,), 1)]])
            #
            # modifiers.append(["small to wide extend gestures", 0, 0, [((13, 14), (1, 2), 3), ]])
            # modifiers.append(["wide to small extend gestures", 0, 0, [((13, 14), (3, 2), 1), ]])
            # modifiers.append(["only wide extend gestures", 0, 0, [((13, 14), (0, 1, 2, 3), 3), ]])
            # modifiers.append(["only small extend gestures", 0, 0, [((13, 14), (0, 1, 2, 3), 1), ]])

            for idxx in [1]:
                vid_idx = idxx
                for mod in modifiers:
                    visualize_name = mod[0]
                    for clip_idx in range(n_clips):
                        clip_poses = clips[clip_idx]['skeletons_3d']
                        clip_audio = clips[clip_idx]['audio_raw']
                        clip_words = clips[clip_idx]['words']
                        clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]
                        clip_anno = clips[clip_idx]['saga_annotation']
                        clip = clips[clip_idx]
                        clip_s_f, clip_e_f = clip['start_frame_no'], clip['end_frame_no']
                        clip_s_t, clip_e_t = clip['start_time'], clip['end_time']

                        sequence_length = 2000  # seconds
                        clip_length = len(clip_audio) / 16000

                        fps = round((clip_e_f - clip_s_f) / max(0.1, clip_e_t - clip_s_t), 4)
                        clip_poses = utils.data_utils.resample_pose_seq(clip_poses, clip_time[1] - clip_time[0],
                                                                        args.motion_resampling_framerate)
                        target_dir_vec = utils.data_utils.convert_pose_seq_to_dir_vec(clip_poses)
                        target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
                        target_dir_vec -= mean_dir_vec

                        if target_dir_vec.shape[0] > 0:
                            # synthesize
                            clip_audio = np.asarray(clip_audio)
                            for selected_vi in range(len(clip_words)):  # make start time of input text zero
                                clip_words[selected_vi][1] -= clip_s_f  # start time
                                clip_words[selected_vi][2] -= clip_s_f  # end time
                            if mod[1] is None:
                                clip_words = []

                            for i in tqdm(range(0, max(1,int(clip_length // sequence_length)))):
                                idx = i * sequence_length * 15
                                idx_end = min(int(clip_length*15),(i + 1) * sequence_length * 15)

                                idx_s = i * sequence_length
                                idx_s_end = min(int(clip_length),(i + 1) * sequence_length)

                                idx_25 = i * sequence_length * 25
                                idx_25_end = min(int(clip_length*25),(i + 1) * sequence_length * 25)

                                # sequence audio
                                audio_in = clip_audio[idx_s * 16000:idx_s_end * 16000]

                                # sequence text
                                text_in = []
                                for selected_vi in range(len(clip_words)):  # make start time of input text zero
                                    temp = copy.deepcopy(clip_words[selected_vi])
                                    temp[1] -= idx_25
                                    temp[2] -= idx_25
                                    text_in.append(temp)

                                # sequence anno
                                if clip_anno is not None:
                                    anno_in = clip_anno[idx_25:idx_25_end + 100]
                                else:
                                    anno_in = None

                                # target sequence
                                target_in = target_dir_vec[idx:idx_end]
                                print(target_in.min(),target_in.max())

                                seed_seq = target_in[:4]

                                reproducibility(0)
                                out_dir_vec, anno_out = generate_gestures(args, generator, lang_model, audio_in, text_in, annotation_data=anno_in, vid=vid_idx, seed_seq=seed_seq, fade_out=False, mean_dir_vec=mean_dir_vec, embedding=embedding, fps=fps, modifier=mod)
                                # make a video
                                sentence_words = []
                                for word, _, _ in text_in:
                                    sentence_words.append(word)
                                sentence = ' '.join(sentence_words)

                                os.makedirs(save_path, exist_ok=True)

                                filename_prefix = '{}_{}_{}'.format(vid, vid_idx, clip_idx)
                                aux_str = '({}, time: {}-{})'.format(vid, str(datetime.timedelta(seconds=clip_time[0])),
                                                                     str(datetime.timedelta(seconds=clip_time[1])))

                                save_folder_name = 'test_save'
                                if not os.path.exists(save_folder_name):
                                    os.makedirs(save_folder_name)

                                write_cont = str(save_counter) + '  ' + sentence + '  ' + aux_str + '\n'
                                handle.write(write_cont)

                                utils.train_utils.create_video_and_save(
                                    sub_folder_save_name, 1, "", str(save_counter),
                                    target_in, out_dir_vec[:target_in.shape[0]], mean_dir_vec,
                                    sentence, audio=audio_in, aux_str=aux_str, withAnnotations=True,save_gen=False, anno=anno_out, video_filename='my_' + str(int(key)) + '_{0:03d}_{1:03d}_person idx-{2:03d} - '.format(clip_idx, i, vid_idx) + visualize_name + "_study_ground")
                                del out_dir_vec
                                save_counter += 1
                                n_saved += 1
    count_bar.close()
    handle.close()


def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)


if __name__ == '__main__':
    reproducibility(0)

    _args = parse_args()
    args = _args
    mode = "from_db_clip"

    option = None
    if len(sys.argv) > 3:
        option = sys.argv[3]

    set_logger()
    for data_id in range(1):
        main(args, mode, None, option, data_id)
