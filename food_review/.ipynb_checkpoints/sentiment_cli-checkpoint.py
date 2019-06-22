import click
from predict import modelpredict

"""
@click.group()
def sentiment_cli():
    pass
"""

@click.command()
@click.argument('text', help="sentiment statement or a text file with of sentiments")
@click.option('--file', default)
def predict(text):
    pred = modelpredict(text)
    print(pred)
    
    

"""
@click.command()
@click.option('--experiment_id', default=0, help="Set the experiment id need to track your train ")
@click.option('--run_name', help="the name of the run so that we could keep track of the run")
@click.option('--x_train')
@click.option('--x_test')
@click.option('--y_train')
@click.option('--y_test')
def train(experiment_id, run_name, xtrain, xtest, ytrain, ytest):
    modeltrain(experiment_id, run_name, xtrain, xtest, ytrain, ytest)
 
    
@click.command()
@click.option(--metrics, type=click.choice(['all','accuracy', 'auc_score', 'f1_score']))
def metric():
    if metrics=all::
        print(client.get_run().get_metrics())
    elif metrics=all:
        print(client.get_run().get_metrics())
    elif metrics=all:
         print(client.get_run().get_metrics()['accuracy'])
    elif auc_score:
        print(client.get_run().get_metrics()['auc_score'])
    else:
        print(client.get_run().get_metrics()['auc_score'])
        """

if __name__ == '__main__':
    sentiment_cli()
    
