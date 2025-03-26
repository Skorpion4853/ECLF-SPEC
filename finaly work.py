def emotion_recognition(audio, json):
    #this function returned flag to all audio
    from TextRecogModel import text_recognition
    from AudioModel import audio_recognition
    from json2list import json_to_list

    texts = json_to_list(json)
    if text_recognition(texts, 0.2) + audio_recognition(audio, 0.8) > -0.5:
        return "Green Light"
    else:
        return "Red Light"
print(emotion_recognition("test/test6.wav", "test/test6.json"))