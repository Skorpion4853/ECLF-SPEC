import os

from yaml import warnings

from TextRecogModel import text_recognition
from AudioModel import audio_recognition, solo_audio_recognition
from json2list import json_to_list

import warnings


def emotion_recognition(audio, json, gl=False):
    #this function returned flag to all audio
    texts = json_to_list(json)
    if gl:
        rubert, lst_emotion = text_recognition(texts, 0.5, gl=True)
        if  rubert + audio_recognition(audio, 0.5) > -0.75:
            return "Green Light", lst_emotion
        else:
            return "Red Light", lst_emotion
    else:
        rubert = text_recognition(texts, 0.6)
        audioR = audio_recognition(audio, 0.4)
        if  rubert + audioR > -0.75:
            return "Green Light"
        else:
            return "Red Light"

def solo_emotion_recognition(audio, text, mode="two model"):
    # mode can insert two model / audio model / text model
    # this param need to select mode for fragment recognition
    if mode == "two model":
        rubert = text_recognition(text, 0.6)
        audioR = solo_audio_recognition(audio)*0.4
        if rubert + audioR > -0.75:
            return rubert, audioR, {"emotions": "Green Light"}
        else:
            return rubert, audioR, {"emotions": "Red Light"}
    elif mode == "audio model":
        rubert = text_recognition(text, 0.6)
        audioR, pred = solo_audio_recognition(audio, mode=True)
        return rubert, audioR*0.4, {"emotions": pred}
    elif mode == "text model":
        rubert, lst = text_recognition(text, 0.6, gl=True)
        audioR = solo_audio_recognition(audio) * 0.4
        return rubert, audioR, lst[0]
    else:
        warnings.warn("This mode is not found")


audio_lst = []
text_lst = []
audios = os.listdir("test/files")
for audio in audios:
    rubert, audioR, result = solo_emotion_recognition(audio, "my tet", mode="text model")
    text_lst.append(rubert)
    audio_lst.append(audioR)
if sum(text_lst) / len(text_lst) + sum(audio_lst) / len(audio_lst) > -0.75:
    print("Green Light")
else:
    print("Red Light")
