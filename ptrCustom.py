from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from inputs_like_api import copy

model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment_custom(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    for tensor in probabilities:
        if tensor[0].detach().item() + tensor[1].detach().item() >= 0.65:
            return "Red Light"
        else:
            return "Green Light"


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


#Вызов функции для проверочного текста
for text in texts:
    out = predict_sentiment_custom([text])
    print(f"text: {text}\nSentiment: {out}\n")




api_custom = copy()
#Функция для ввода апи формат данных
def make_flags(api_input):
    for i in range(len(api_input)):
        emotion = predict_sentiment_custom(api_input[i]['text'])
        api_input[i].update({'emotions': emotion})
    return api_input

print(make_flags(api_custom))

