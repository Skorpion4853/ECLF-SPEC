def predict(file):
    #this function returned count who clf all sound
    import torch
    from aniemore.recognizers.voice import VoiceRecognizer
    from aniemore.models import HuggingFaceModel
    count = 0
    model = HuggingFaceModel.Voice.WavLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vr = VoiceRecognizer(model=model, device=device)
    pred = vr.recognize(file, return_single_label=True)
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
    from pydub import AudioSegment
    import warnings

    if file.endswith('.aac'):
        wav_audio = AudioSegment.from_file(file, format="aac")
        file = file.replace('aac', 'wav')
        wav_audio.export(file, format="wav")
        return predict(file) * weight

    elif file.endswith('.wav'):
        return predict(file) * weight

    else:
        warnings.warn('This is none available format', FutureWarning)
