from pycparser.ply.yacc import resultlimit


def emotion_recognition(audio, json, gl=False):
    #this function returned flag to all audio
    from TextRecogModel import text_recognition
    from AudioModel import audio_recognition
    from json2list import json_to_list

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