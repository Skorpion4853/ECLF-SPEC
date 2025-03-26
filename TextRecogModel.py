def text_recognition(texts, weight):
    #this function using rubert to clf transcription text return sum all predict * weight
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    count = 0
    model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()

    def get_sentiment(text, return_type='label'):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
            proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
        if return_type == 'label':
            return model.config.id2label[proba.argmax()]
        elif return_type == 'score':
            return proba.dot([-1, 0, 1])
        return proba
    for text in texts:
        try:
            if get_sentiment(text, 'label') == 'negative':
                count -= 1
            elif get_sentiment(text, 'label') == 'neutral':
                count += 0
            elif get_sentiment(text, 'label') == 'positive':
                count += 1
        except ValueError:
            continue
    try:
        return count/len(texts) * weight
    except:
        #I forgot error(
        return count * weight