from mlflow.tracking import MlflowClient
from PIL import Image  


def modelmetrics(metric:str) -> str:
    
    '''
    This function takes metricname and return score value:
    The availalible matrics:
    
    all : All metrics used to evaluate the model
    accuracy:
    auc_score:
    f1_score: 
    
    
    Parameters:
    ----------
    metric: str
    
    Return:
    ------
    str
    '''
    client = MlflowClient()
    
    try:
        if metric =='all':
            return client.get_run('3e8f282376364196a439678a824bccf8').data.metrics
        
        elif metric=='accuracy':
            return f'Accuracy: {client.get_run("3e8f282376364196a439678a824bccf8").data.metrics["Accuracy"]}'
    
        elif metric=='auc_score':
            return f'AUC Score: {client.get_run("3e8f282376364196a439678a824bccf8").data.metrics["auc_score"]}'
        
        elif metric=='f1_score':
            return f'FI Score: {client.get_run("3e8f282376364196a439678a824bccf8").data.metrics["f1_score"]}'
        
        else:
            return client.get_run('3e8f282376364196a439678a824bccf8').data.metrics
        
    except ValueError as e:
        return f'ValueError is {e}'
    
    
    
def metricsvisualizer(chart):
    '''
    The function takes one of the images listed below and print it to the console.
    chart:
    1. classification_report
    2. classification_error
    3. confusion_matrix
    4. roc_auc
    
    '''
    
    try:
        if chart=='classification_report':
            return Image.open("image/classification_report.png")
        elif chart =='classification_error':
            return Image.open("image/ClassificationError.png")
        elif chart == 'confusion_matrix':
            return Image.open("image/confusionmatrix.png")
        elif chart == 'roc_auc':
            return Image.open('image/rocauc.png')
        
    except ValueError as e:
        return f'ValueError is {e}'
    
 
        