from sklearn.external import joblib
import pandas as pd


def modelpredict(text:str):
    try:
        model = joblib.load('pipeline_lr.pkl')
        if not text.endswith('txt'):
            string = [text]
            preds = model.predict(string)[0]
            pred_prb = round(model.predict_proba(string)[:,1][0], 2)
            return f'Verbatism: {text}\n Sentiment Value: {preds}\n Sentiment score: {pred_prb}%'
        else:
            df = pd.read_csv(text, names =['sentiment'], delimiter='\t', encoding='utf-8')
            df['predictions'] = model.predict(df['sentiment'])
            score = model.predict_proba(df['sentiment'])
            sentiment_score = [i[1] for i in score]
            df['sentiment_score'] = sentiment_score
            return df
    except ValueError as e:
        return f'Value error is {e}'