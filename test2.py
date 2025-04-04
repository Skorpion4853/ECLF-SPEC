from transformers import pipeline
model = pipeline(model="r1char9/rubert-base-cased-russian-sentiment")
def text_recognition(text, weight):
    result = model(text)
    if result[0]["label"] == "negative":
        return -1 * weight
    elif result[0]["label"] == "positive":
        return 1 * weight
    elif result[0]["label"] == "neutral":
        return 0 * weight