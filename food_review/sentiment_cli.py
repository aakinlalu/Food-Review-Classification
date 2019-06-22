import click
from .predict import modelpredict
from .metrics import modelmetrics, metricsvisualizer


@click.group()
def sentiment_cli():
    pass


@click.command()
@click.option('--text', help="it can be single text review or text file with review with .txt/.csv")
def predict(text):
    pred = modelpredict(text)
    click.echo(pred)
    
 
@click.command()
@click.option('--metric', default='all', type=click.Choice(['all','accuracy', 'auc_score', 'f1_score']), help="Select which metrics you want to see")
def metrics(metric):
    metrics = modelmetrics(metric)
    click.echo(metrics)
    
    
    
@click.command()
@click.option('--chart', default='confusion_matrix', type=click.Choice(['confusion_matrix','classification_report', 'classification_error', 'roc_auc']))
def metricsvisualizer(chart):
    chart = metricsvisualizer(chart)
    click.echo(chart)
    
sentiment_cli.add_command(predict)
sentiment_cli.add_command(metrics)
sentiment_cli.add_command(metricsvisualizer)
    

if __name__ == '__main__':
    sentiment_cli()
    
