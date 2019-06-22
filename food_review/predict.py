from sklearn.externals import joblib
import pandas as pd


def modelpredict(text:str):
    """
    The function takes a single review statement or text file of reviews and predict sentiment.
    
    Parameters:
    ----------
    text: str
    
    Returns
    -------
    df: pd.DataFrame or str
    """
    try:
        model = joblib.load('pipeline_lr.pkl')
        if text.endswith('txt'):
            df = pd.read_csv(text, names =['sentiment'], delimiter='\t', encoding='utf-8')
            df['predictions'] = model.predict(df['sentiment'])
            score = model.predict_proba(df['sentiment'])
            sentiment_score = [i[1] for i in score]
            df['sentiment_score'] = sentiment_score
            return df
        elif text.endswith('csv'):
            df = pd.read_csv(text, names =['sentiment'], delimiter='\t', encoding='utf-8')
            df['predictions'] = model.predict(df['sentiment'])
            score = model.predict_proba(df['sentiment'])
            sentiment_score = [i[1] for i in score]
            df['sentiment_score'] = sentiment_score
            return df
        else:
            string = [text]
            preds = model.predict(string)[0]
            pred_prb = round(model.predict_proba(string)[:,1][0], 2)
            return f'Verbatism: {text}\n Sentiment Value: {preds}\n Sentiment score: {pred_prb}%'
    except ValueError as e:
        return f'Value error is {e}'