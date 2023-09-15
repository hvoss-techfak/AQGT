import logging
import os
import pickle
import traceback

import lmdb
import pyarrow
from model.vocab import Vocab
from tqdm import tqdm


def build_vocab(name, dataset_list, cache_path, word_vec_path=None, feat_dim=None):
    logging.info('  building a language model...')
    if not os.path.exists(cache_path):
        lang_model = Vocab(name)
        print(cache_path)
        for dataset in dataset_list:
            logging.info('    indexing words from {}'.format(dataset.lmdb_dir))
            index_words(lang_model, dataset.lmdb_dir)

        if word_vec_path is not None:
            print("loading word vector fasttext")
            lang_model.load_word_vectors(word_vec_path, feat_dim)
        else:
            print("not fasttext vector found!")
            raise Exception

        with open(cache_path, 'wb') as f:
            pickle.dump(lang_model, f)
    else:
        logging.info('    loaded from {}'.format(cache_path))
        with open(cache_path, 'rb') as f:
            lang_model = pickle.load(f)
        logging.info('    indexed %d words' % lang_model.n_words)
        if word_vec_path is None:
            lang_model.word_embedding_weights = None
            assert False
        elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
            logging.warning('    failed to load word embedding weights. check this')
            assert False

    return lang_model


def index_words(lang_model, lmdb_dir):
    lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    txn = lmdb_env.begin(write=False)
    cursor = txn.cursor()

    for key, buf in tqdm(cursor):
        try:
            video = pyarrow.deserialize(buf)

            for clip in video['clips']:
                for word_info in clip['words']:
                    word = word_info[0]
                    lang_model.index_word(word)
        except pyarrow.lib.ArrowInvalid:
            pass
        except pyarrow.lib.ArrowIOError:
            pass
        except Exception:
            traceback.print_exc()

    lmdb_env.close()
    logging.info('    indexed %d words' % lang_model.n_words)

    # filtering vocab
    # MIN_COUNT = 3
    # lang_model.trim(MIN_COUNT)
