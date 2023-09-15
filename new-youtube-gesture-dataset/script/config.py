# ------------------------------------------------------------------------------
# Copyright (c) ETRI. All rights reserved.
# Licensed under the BSD 3-Clause License.
# This file is part of Youtube-Gesture-Dataset, a sub-project of AIR(AI for Robots) project.
# You can refer to details of AIR project at https://aiforrobots.github.io
# Written by Youngwoo Yoon (youngwoo@etri.re.kr)
# ------------------------------------------------------------------------------

from datetime import datetime


class Config:
    DEVELOPER_KEY = ""  # your youtube developer id
    OPENPOSE_BASE_DIR = "/mnt/work/work/openpose/"
    OPENPOSE_BIN_PATH = "build/examples/openpose/openpose.bin"


class TEDConfig(Config):
    YOUTUBE_CHANNEL_ID = "UCAuUUnT6oDeKwE6v1NGQxug"
    WORK_PATH = 'dataset/AQGT/videos/'
    CLIP_PATH = WORK_PATH + "/clips"
    VIDEO_PATH = WORK_PATH + "/withVTT"
    SKELETON_PATH = WORK_PATH + "/withVTT"
    SUBTITLE_PATH = VIDEO_PATH
    OUTPUT_PATH = WORK_PATH + "/output"
    VIDEO_SEARCH_START_DATE = datetime(2011, 3, 1, 0, 0, 0)
    LANG = 'en'
    SUBTITLE_TYPE = 'auto'
    FILTER_OPTION = {"threshold": 0.02}
    OVERWRITE_FILTERING = False
    FILETYPE = '.webm'



class TED2Config(Config):
    YOUTUBE_CHANNEL_ID = "UCAuUUnT6oDeKwE6v1NGQxug"
    WORK_PATH = 'dataset/AQGT/videos/'
    CLIP_PATH = WORK_PATH + "/clips"
    VIDEO_PATH = WORK_PATH + "/videos"
    SKELETON_PATH = WORK_PATH + "/videos"
    SUBTITLE_PATH = VIDEO_PATH
    OUTPUT_PATH = WORK_PATH + "/out"
    VIDEO_SEARCH_START_DATE = datetime(2011, 3, 1, 0, 0, 0)
    LANG = 'en'
    SUBTITLE_TYPE = 'auto'
    FILTER_OPTION = {"threshold": 0.00}
    OVERWRITE_FILTERING = True
    FILETYPE = '.webm'

class SaGAConfig(Config):
    YOUTUBE_CHANNEL_ID = "UCAuUUnT6oDeKwE6v1NGQxug"
    WORK_PATH = 'dataset/SaGA/videos/'
    CLIP_PATH = WORK_PATH + "/clips"
    VIDEO_PATH = WORK_PATH + "/SaGA1"
    SKELETON_PATH = WORK_PATH + "/SaGA1"
    SUBTITLE_PATH = VIDEO_PATH
    OUTPUT_PATH = WORK_PATH + "/output"
    VIDEO_SEARCH_START_DATE = datetime(2011, 3, 1, 0, 0, 0)
    LANG = 'en'
    SUBTITLE_TYPE = 'saga'
    FILTER_OPTION = {"threshold": 0.00}
    OVERWRITE_FILTERING = False
    FILETYPE = '.mp4'

class SaGA_VAL_Config(Config):
    YOUTUBE_CHANNEL_ID = "UCAuUUnT6oDeKwE6v1NGQxug"
    WORK_PATH = 'dataset/SaGA/videos/'
    CLIP_PATH = WORK_PATH + "/video"
    VIDEO_PATH = WORK_PATH + "/video"
    SKELETON_PATH = WORK_PATH + "/video"
    SUBTITLE_PATH = VIDEO_PATH
    OUTPUT_PATH = WORK_PATH + "/output"
    VIDEO_SEARCH_START_DATE = datetime(2011, 3, 1, 0, 0, 0)
    LANG = 'en'
    SUBTITLE_TYPE = 'saga'
    FILTER_OPTION = {"threshold": 0.00}
    OVERWRITE_FILTERING = False
    FILETYPE = '.mp4'

# SET THIS
my_config = SaGA_VAL_Config
