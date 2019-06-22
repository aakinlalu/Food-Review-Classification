FOOD REVIEW CLASSIFICATION CLI APPLICATION
---
Build classification model to classifier food review sentiments using logistic regression, naive bayes through sklearn and mlflow.    


       |--food_review
       |      |--data/
       |      |
       |      |--image/
       |      |
       |      |--mlruns
       |      |
       |      |--metrics.py
       |      |
       |      |--pipeline.pkl
       |      |
       |      |--predict.py
       |      |
       |      |--sentiment_cli.py
       |      |
       |      |--train.py
       |
       |--README.md
       |
       |--requirment.txt
       |
       |--setup.py
       

### What does the application do?
The Review Sentiment is a supervised machine learning cli application that identifies if a review has postive (1) or negative (0) sentiment with probability score.

### How to install it
From the project directory in terminal:  
1. create python enviroment using conda or venv
2. ***pip install -r requirement.txt***

### Example Dev Usage
1. from RUBIX directory cd food_review
2. Try to use help undertand the cli  
   ***python sentiment_cli.py --help***
   
   Usage: sentiment_cli.py [OPTIONS] COMMAND [ARGS]...

   Options:
      --help  Show this message and exit.

   Commands:  
     metrics  
     metricsvisualizer  
     predict  
     
3. To use it to predict: 
    1. Try with predict help:  
    
        ***python sentiment_cli.py  predict --help***
        
        Usage: sentiment_cli.py predict [OPTIONS]

       Options:
       
       --text TEXT  it can be single text review or text file with review with
               .txt/.csv
       --help       Show this message and exit.
       
    2.  Try to use predict "I do not like rice anymore":  
    
          ***python sentiment_cli.py predict --text="I do not like rice anymore"***
          
          Verbatism: I do not like rice anymore
          Sentiment Value: 0
          Sentiment score: 0.42%
          
4. To check the metrics of the model:
   1. Check out help:  
      ***python sentiment_cli.py metrics --help***
      
      Usage: sentiment_cli.py metrics [OPTIONS]

       Options:
         --metric [all|accuracy|auc_score|f1_score]
                                  Select which metrics you want to see
         --help                          Show this message and exit.
   2. Check all the metrics: 
   
       ***python sentiment_cli.py metrics --metric=all***
       
       {'auc_score': 0.76, 'Accuracy': 0.76, 'f1_score': 0.75}

### Build it
from Food-Review-Classification directory:  
    ***python setup.py develop***
