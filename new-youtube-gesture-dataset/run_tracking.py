# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import bz2
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

import traceback
from AlphaPoseHelper import AlphaPoseHelper, RealtimeSequence
from config import my_config


def doThread(file):
    try:
        if not os.path.exists(file + "_calc.kp"):
            if not os.path.exists(file + ".lock"):
                f = open(file + ".lock", "a")
                f.write("lock")
                f.close()
                print(file)
                a = AlphaPoseHelper(file)
                ret = RealtimeSequence(alphaPoseHelper=a)
                seq_in = []
                for out in tqdm(a,total=a.det_loader.datalen):
                    try:
                        frameP = ret.newFrame(out, store_image=False)
                        seq_in.append(frameP)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        seq_in.append(None)
                pickle.dump(seq_in, bz2.BZ2File(file + "_calc.kp", 'wb'))
                del a
                del ret

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
    finally:
        import torch
        torch.cuda.empty_cache()
        if os.path.exists(file + ".lock"):
            os.remove(file + ".lock")

if __name__ == '__main__':
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    folder = my_config.VIDEO_PATH
    vv = [folder + "/" + f for f in os.listdir(folder + "/")]

    for f in vv:
        if f.endswith(my_config.FILETYPE):
            doThread(f)

