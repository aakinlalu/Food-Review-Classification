from setuptools import setup, find_packages

setup(
      name="food_review",
      version="0.1",
      author_email="aakinlalu@outlook.com",
      packages=find_packages('food_review'),
      package_dir={'':'food_review'},
      include_package_data=True,
      install_requires=['scikit-learn', 'yellowbrick', 'mlflow', 'seaborn', 'numpy', 'pandas', 'matplotlib', 'Click'],
      package_data={ 
                     '': ['*.pkl'],
                     'mlruns': ['0/3e8f282376364196a439678a824bccf8/*'],
                     'data': ['*.txt'],
                   },
      entry_points={
        'console_scripts': ['food_review = food_review.sentiment_cli:sentiment_cli']
      }
        
)