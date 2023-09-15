import logging
import os
import pickle
import socket
import traceback

import librosa
import lmdb as lmdb
import lz4.frame
import numpy as np
import pyarrow
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, BertTokenizer, BertModel
from transformers.utils import PaddingStrategy

from scripts.data_loader.data_preprocessor import DataPreprocessor
from scripts.model.vocab import Vocab
from scripts.utils.data_utils import calc_spectrogram_length_from_motion_length, make_audio_fixed_length

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def proc_audio(audio, audio_numpy):
    audio_flag = (audio > 1e-3).float()
    audio_thresh = (audio * audio_flag).sum() / audio_flag.sum()

    audio_len = 340 * 3
    ind = torch.arange(audio_len)
    au_size = audio.size(0)
    audio_step = au_size // audio_len
    ind_step = ind * audio_step
    sample_bias = audio_step // 2
    ind_step_bias = ind_step + sample_bias
    ind_step_bias[-1] = au_size - 1 if ind_step_bias[-1] >= au_size else ind_step_bias[-1]
    audio_sample = audio.abs()[ind_step_bias]
    audio_flag = (audio_sample > audio_thresh).float()

    hop_len = au_size // audio_len
    sr_num = 16000
    oenv = librosa.onset.onset_strength(y=audio_numpy, sr=sr_num, hop_length=hop_len)
    start_skip = (oenv.size - audio_len) // 2
    audio_oenv = torch.from_numpy(oenv)[start_skip:audio_len + start_skip]

    audio_beat = torch.stack([audio_flag, audio_oenv])

    return audio_beat


def word_seq_add_beat_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    beats, word_seq, text_padded, poses_seq, vec_seq, audio, audio_var_bottom, audio_var_top, spectrogram, aux_info, anno = zip(
        *data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    # word_seq = pad_sequence(word_seq, batch_first=True).long()

    beats = default_collate(beats)
    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    audio_var_bottom = default_collate(audio_var_bottom)
    audio_var_top = default_collate(audio_var_top)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    # print(anno)
    if anno[0] is not None:
        annotations = [default_collate(an) for an in anno]
    else:
        annotations = torch.zeros((34,)) - 1

    return beats, word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, audio_var_bottom, audio_var_top, spectrogram, aux_info, annotations


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    _, word_seq, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info


def default_collate_fn(data):
    _, text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info


class SpeechMotionDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec, beat_path,
                 speaker_model=None, remove_word_timing=False, save_flag=False, eval_mode=False, beat_generating=False,
                 pretrained_lang=None):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_dir_vec = mean_dir_vec
        self.remove_word_timing = remove_word_timing

        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = calc_spectrogram_length_from_motion_length(
            n_poses, pose_resampling_fps)

        self.lang_model = None
        self.save_flag = save_flag

        print(socket.gethostname())

        print("creating dummy language model")

        self.pret_path = pretrained_lang
        self.embedding = None


        self.beat_path = beat_path
        map_size = 1024 * 150  # in MB
        map_size <<= 20  # in B

        from scripts.model.VQVAE_2_audio import VQ_VAE_2_audio

        self.audio_vqvae = VQ_VAE_2_audio.load_from_checkpoint("pretrained/vqvae_audio/vqvae_audio.ckpt",
                                                               strict=False).eval().cuda()

        self.beat_caching = beat_generating
        self.setBeatProducing(self.beat_caching)

        self.ie = 0
        os.makedirs(self.beat_path, exist_ok=True)
        print("init sqlite")
        logging.info("Reading data '{}'...".format(lmdb_dir))
        logging.info("Preload_dir: " + lmdb_dir + '_cache')
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            logging.info('Creating the dataset cache...')
            assert mean_dir_vec is not None
            if mean_dir_vec.shape[-1] != 3:
                mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))

            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec)
            data_sampler.run()
        else:
            logging.info('Found the cache {}'.format(preloaded_dir))

        # init lmdb
        self.preloaded_dir = preloaded_dir
        self.map_size = map_size
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False, map_size=map_size)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']
        with self.lmdb_env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        self.lmdb_env.close()
        del self.lmdb_env
        self.lmdb_env = None

        self.Wav2Vec2processor = None

        self.tokenizer = None

        # make a speaker model
        if speaker_model is None or speaker_model == 0:
            precomputed_model = lmdb_dir + '_speaker_model.pkl'
            if not os.path.exists(precomputed_model):
                self._make_speaker_model(lmdb_dir, precomputed_model)
            else:
                with open(precomputed_model, 'rb') as f:
                    self.speaker_model = pickle.load(f)
        else:
            self.speaker_model = speaker_model

    def setBeatProducing(self, enabled):
        if not enabled:
            self.lmdb_beat_env = lmdb.open(self.beat_path, lock=False, readonly=True)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.Wav2Vec2processor is None:
            self.Wav2Vec2processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.Wav2Vec2model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").cuda()
            self.Wav2Vec2model.eval()
            for p in self.Wav2Vec2model.parameters():
                p.requires_grad = False
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained("bert-base-uncased").cuda()
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.preloaded_dir, readonly=True, lock=False, map_size=self.map_size)
        with self.lmdb_env.begin(write=False) as txn:
            key = self.keys[idx]
            sample = txn.get(key)
            sample = pyarrow.deserialize(lz4.frame.decompress(sample))
            anno = None
            if len(sample) == 6:
                word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
            else:
                word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info, anno = sample

        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            if self.remove_word_timing:
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        n_words += 1
                space = int(n_frames / (n_words + 1))
                for i in range(n_words):
                    idx = (i + 1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[i][0])
            else:
                prev_idx = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        duration = aux_info['end_time'] - aux_info['start_time']
        do_clipping = True

        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
            audio = make_audio_fixed_length(audio, self.expected_audio_length)
            vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
        else:
            sample_end_time = None

        # word_list = []
        word_list = [x[0] for x in word_seq]
        start_time_list = [str(x[1]) for x in word_seq]
        end_time_list = [str(x[2]) for x in word_seq]
        sent = ' '.join(word_list)

        rec_sent = '|'.join(word_list)
        rec_start = '|'.join(start_time_list)
        rec_end = '|'.join(end_time_list)
        word_raw = rec_sent + ' ' + rec_start + ' ' + rec_end

        aux_info['word_list'] = sent
        aux_info['word_raw'] = word_raw

        extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)

        vec_seq_ = torch.from_numpy(vec_seq.copy())
        vec_seq_ = vec_seq_.reshape((vec_seq_.shape[0], -1)).float()
        pose_seq = torch.from_numpy(pose_seq.copy()).reshape((pose_seq.shape[0], -1)).float()
        audio_numpy = audio.copy().astype(np.float32)
        audio = torch.from_numpy(audio.copy()).float()
        if anno is not None:
            anno = [torch.from_numpy(an.copy()).float() for an in anno]

        aux_key = '_'.join([aux_info['vid'],
                            str(aux_info['start_frame_no']),
                            str(aux_info['end_frame_no'])])

        k = (aux_key + '.pt').encode('utf-8', 'ignore')

        if self.beat_caching:
            aux_info['beat_k'] = k
            audio_beat = proc_audio(audio, audio_numpy)
        else:
            try:
                with self.lmdb_beat_env.begin(write=False) as txn:
                    k = (aux_key + '.pt').encode('utf-8', 'ignore')
                    beat = txn.get(k)
                    audio_beat = torch.from_numpy(pyarrow.deserialize(beat))
            except Exception:
                traceback.print_exc()
                audio_beat = proc_audio(audio, audio_numpy)
        spectrogram = torch.from_numpy(np.zeros((1,)))

        in_audio = audio.reshape(1, -1)
        in_audio -= in_audio.min()
        in_audio /= max(0.0000001, in_audio.max()) / 2
        in_audio -= 1
        in_audio = in_audio[:, :int((in_audio.shape[1] // 4000) * 4000)].reshape((in_audio.shape[0], -1, 4000))

        with torch.no_grad():
            audio_var_bottom = torch.zeros((in_audio.shape[0], in_audio.shape[1], 16 * 16), device=in_audio.device)
            audio_var_top = torch.zeros((in_audio.shape[0], in_audio.shape[1], 8 * 8), device=in_audio.device)
            for i in range(in_audio.shape[1]):
                in_data = in_audio[:, i, :]
                in_temp = torch.zeros((in_data.shape[0], 4096), device=in_audio.device)
                in_temp[:, :4000] = in_data
                in_temp = torch.moveaxis(in_temp.reshape((in_data.shape[0], 64, 64, 1)), 3, 1)

                quant_t, quant_b, diff_a, _, _ = self.audio_vqvae.generator.encode(in_temp.cuda())
                quant_b = quant_b.reshape(audio_var_bottom.shape[0], 16 * 16).cpu()
                quant_t = quant_t.reshape(audio_var_top.shape[0], 8 * 8).cpu()

                audio_var_bottom[:, i, :] = quant_b
                audio_var_top[:, i, :] = quant_t

            audio_var_bottom = audio_var_bottom.flatten()
            audio_var_top = audio_var_top.flatten()

        with torch.no_grad():
            ad1 = self.Wav2Vec2processor(audio.reshape(1, -1).cuda(), sampling_rate=16000, return_tensors="pt")
            input_values = ad1.input_values.squeeze(0).cuda()

            log = self.Wav2Vec2model(input_values=input_values).logits

            encoded_input = self.tokenizer(aux_info['word_list'], return_tensors='pt', padding=PaddingStrategy.LONGEST,
                                           max_length=500, truncation=True, )
            encoded_input.data["input_ids"] = encoded_input.data["input_ids"].cuda()
            encoded_input.data["token_type_ids"] = encoded_input.data["token_type_ids"].cuda()
            encoded_input.data["attention_mask"] = encoded_input.data["attention_mask"].cuda()

            output = self.model(**encoded_input)["pooler_output"]

        return audio_beat, output.detach().cpu(), extended_word_seq, pose_seq, vec_seq_, log.detach().cpu(), audio_var_bottom, audio_var_top, spectrogram, aux_info, anno

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model

    def _make_speaker_model(self, lmdb_dir, cache_path):
        logging.info('  building a speaker model...')
        speaker_model = Vocab('vid', insert_default_tokens=False)

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        txn = lmdb_env.begin(write=False)
        cursor = txn.cursor()
        for key, value in tqdm(cursor):
            try:
                video = pyarrow.deserialize(value)
                vid = video['vid']
                speaker_model.index_word(vid)
            except pyarrow.lib.ArrowInvalid:
                pass
            except pyarrow.lib.ArrowIOError:
                pass
            except Exception:
                traceback.print_exc()

        lmdb_env.close()
        logging.info('    indexed %d videos' % speaker_model.n_words)
        self.speaker_model = speaker_model

        # cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.speaker_model, f)


def combine_speech_vocab_model(args, models, output_path, fill_word_vectors=False):
    speaker_model = Vocab('vid', insert_default_tokens=False)
    for precomputed_model in models:
        with open(precomputed_model, 'rb') as f:
            temp_model = pickle.load(f)
            speaker_model.add_vocab(temp_model)
    if fill_word_vectors:
        speaker_model.load_word_vectors(args.wordembed_path, args.wordembed_dim)
    with open(output_path, 'wb') as f:
        pickle.dump(speaker_model, f)


def fill_embedding_vocab(model_path, args):
    with open(model_path, 'rb') as f:
        temp_model = pickle.load(f)
    if args.wordembed_path is not None:
        temp_model.load_word_vectors(args.wordembed_path, args.wordembed_dim)
    with open(model_path, 'wb') as f:
        pickle.dump(temp_model, f)


if __name__ == '__main__':
    pass
