mport numpy as np
import pandas as pd
import warnings
import logging
import sys

import seaborn as sns
sns.set_context("poster")
sns.set(rc={'figure.figsize': (10, 20.)})
#sns.set_style("whitegrid")

import plotly_express as px

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)

import sklearn
import mlflow
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import mlflow.sklearn


from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC, ClassPredictionError
from sklearn.externals import joblib
import matplotlib.pyplot as plt


xtrain, xtest, ytrain, ytest = train_test_split(train_df['sentiment'],train_df['target'], test_size=0.3, random_state=42)

def train(experiment_id, run_name, xtrain, xtest, ytrain, ytest):
    
    np.random.seed(100)

    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        
        tfid_vect =TfidfVectorizer(analyzer='word', tokenizer=nltk.tokenize.word_tokenize, stop_words='english', min_df=5)
        
        
        my_pipeline = Pipeline(steps=[('vectorizer', tfid_vect),
                                       ('lr', LogisticRegression(random_state=42))])
        
           
        my_pipeline.fit(xtrain, ytrain)
        predictions = my_pipeline.predict(xtest)
                                      
        joblib.dump(my_pipeline, 'pipeline_lr.pkl')
        
        accuracy = accuracy_score(ytest, predictions)
        
        f1score = f1_score(ytest, predictions)
        
        auc_score = roc_auc_score(ytest, predictions)
        
        class_report = classification_report(ytest, predictions)
        
        print(f'Accuracy : {round(accuracy, 2)}')
        print(f'f1_score : {round(f1score, 2)}')
        print(f'auc_score : {round(auc_score, 2)}')
        print(f'class_report : \n {class_report}')
        
        mlflow.log_metric('Accuracy', round(accuracy, 2))
        mlflow.log_metric('f1_score', round(f1score, 2))
        mlflow.log_metric('auc_score', round(auc_score, 2))
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
        
        visualizer = ClassificationReport(my_pipeline, ax=ax1, classes=[0,1])
        visualizer.fit(xtrain, ytrain)
        visualizer.score(xtest, ytest)
        a=visualizer.poof(outpath="image/classification_report.png")
        print(' ')
        
        mlflow.log_artifact("image/classification_report.png")
        
        # The ConfusionMatrix visualizer taxes a model
        cm = ConfusionMatrix(my_pipeline, ax=ax2, classes=[0,1])
        cm.fit(xtrain, ytrain)
        cm.score(xtest, ytest) 
        b=cm.poof(outpath="image/confusionmatrix.png")
        
        mlflow.log_artifact("image/confusionmatrix.png")
        print(' ')
        
        vis = ROCAUC(my_pipeline, ax=ax3, classes=[0,1])
        vis.fit(xtrain, ytrain)  # Fit the training data to the visualizer
        vis.score(xtest, ytest)  # Evaluate the model on the test data
        c = vis.poof(outpath="image/rocauc.png")             # Draw/show/poof the data
        print(' ')
        mlflow.log_artifact("image/rocauc.png")
        
        visual = ClassPredictionError(my_pipeline, ax=ax4, classes=[0,1])
        visual.fit(xtrain, ytrain)
        visual.score(xtest, ytest)
        g = visual.poof(outpath="image/ClassificationError.png")
        print(' ')
        mlflow.log_artifact("image/ClassificationError.png")
        
        
        return run.info.run_uuid