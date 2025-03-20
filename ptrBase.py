from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from inputs_like_api import copy

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment_base(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    lst = probabilities[0].detach().tolist()
    for i in range(len(lst)):
        lst[i] = round(lst[i], 2)
    sentiment_map = {"Very Negative": lst[0], "Negative": lst[1], "Neutral": lst[2], "Positive": lst[3], "Very Positive": lst[4]}
    return sentiment_map
#Текст для проверки
texts = [
    "Вот ваш заказ",
    "Ваша заведение полный мусор",
    "Вот ваш салатик сер",
    "Вот ваш заказ приятного аппетита",
    "Вот ваш салатик приятного аппетита",
    "Вы тут один?",
    "Вы скоро закончите?",
    "Ваш стол на одного проходите",
    "Так, Брестин?",
    "вы хотите Брестин, всё верно?",
    "То есть вы только что харкнули в мой суп?",
    "Иди нахуй со своим заказом",
    "Этот заказ к сожалению невозожно выполнить приносим свои извинения",
    "Это ахуеть какое вкусное блюдо",
    "Я бы хотел пожаловаться на сервис"
]

#Вывод результатов для texts
for text, sentiment in zip(texts, predict_sentiment_base(texts)):
    print(f"Text: {text}\nSentiment: {sentiment}\n")


api_base = copy()
#Функция для ввода апи формат данных
def make_flags(api_input):
    for i in range(len(api_input)):
        emotion = predict_sentiment_base(api_input[i]['text'])
        api_input[i].update({'emotions': emotion})
        return api_input

print(make_flags(api_base))
