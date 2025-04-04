import os

from yaml import warnings

from test2 import text_recognition
from AudioModel import audio_recognition, solo_audio_recognition
from json2list import json_to_list

import warnings

def test(audio, text):
    rubert = text_recognition(text, 0.6)
    audio = solo_audio_recognition(audio, 0.4)
    if rubert+audio >= -0.6:
        return "Green Light"
    else:
        return "Red Light"

