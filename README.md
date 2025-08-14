# the-challenge

## Consider using a venv

```
# Create a virtual environment
python[your python version] -m venv ~/.venvs/py313

# Activate it
source ~/.venvs/py313/bin/activate

# install all required packages
pip install -r requirments
```

## Start experimenting in demo.ipynb
You can use the audusd.zip data here, but do remember to check
```
sentiment_analyzer = Sentiment(
    root_data_dir="your path to audusd.zip",
    stock_code="audusd",
    device=1  # Use GPU
)
```

## Features
- [x] Multi-Step forecasts with confidence intervals and trading signals, using CNN model;
- [x] Sentiment anlaysis using Reddit r/forex comments and sentiment score as a predictive variable;
- [  ] Look into if model is memorizing instead of learning;
- [  ] Add backtesting, using predicted signals; 
- [  ] Automatically partial fit the model when new data comes in, every 2h/1d;

## Acknowledgements
@
