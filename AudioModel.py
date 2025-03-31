import torch
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel
from pydub import AudioSegment
import warnings

model = HuggingFaceModel.Voice.WavLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vr = VoiceRecognizer(model=model, device=device)

def predict(file):
    #this function returned count who clf all sound
    pred = vr.recognize(file, return_single_label=True)
    count = 0
    if pred == "anger" or pred == "disgust":
        count -= 1
    elif pred == "fear" or pred == 'sadness':
        count -= 0.5
    elif pred == 'neutral' or pred == 'enthusiasm':
        count += 0
    elif pred == 'happiness':
        count += 1
    return count


def audio_recognition(file, weight):
    #this function check our file on format and replace format to wav

    if file.endswith('.aac'):
        wav_audio = AudioSegment.from_file(file, format="aac")
        file = file.replace('aac', 'wav')
        wav_audio.export(file, format="wav")
        return predict(file) * weight

    elif file.endswith('.wav'):
        return predict(file) * weight

    else:
        warnings.warn('This is none available format', FutureWarning)

def solo_predict(file, mode):
    # this function returned count who clf all sound
    if mode:
        pred_e = vr.recognize(file, return_single_label=False)
        pred = vr.recognize(file, return_single_label=True)
        if pred == "anger" or pred == "disgust":
            return -1, pred_e
        elif pred == "fear" or pred == 'sadness':
            return -0.5, pred_e
        elif pred == 'neutral' or pred == 'enthusiasm':
            return 0, pred_e
        elif pred == 'happiness':
            return 1, pred_e
    else:
        pred = vr.recognize(file, return_single_label=True)
        if pred == "anger" or pred == "disgust":
            return -1
        elif pred == "fear" or pred == 'sadness':
            return -0.5
        elif pred == 'neutral' or pred == 'enthusiasm':
            return 0
        elif pred == 'happiness':
            return 1

def solo_audio_recognition(file, mode=False):
    if file.endswith('.aac'):
        wav_audio = AudioSegment.from_file(file, format="aac")
        file = file.replace('aac', 'wav')
        wav_audio.export(file, format="wav")
        return solo_predict(file, mode)

    elif file.endswith('.wav'):
        return solo_predict(file, mode)

    else:
        warnings.warn('This is none available format', FutureWarning)