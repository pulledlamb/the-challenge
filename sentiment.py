class MyTextDataset:
    """Custom dataset class for loading text files with timestamps."""
    
    def __init__(self, root_data_dir, stock_code):
        import os
        import zipfile
        
        self.root_data_dir = root_data_dir
        self.stock_code = stock_code

        # Ensure the data is unzipped
        self.check_and_unzip()
        
        self.data_dir = os.path.join(self.root_data_dir, self.stock_code)
        
        # Get the list of files in the unzipped directory
        self.file_list = os.listdir(self.data_dir)
        self.num_samples = len(self.file_list)

    def check_and_unzip(self):
        import os
        import zipfile
        
        # Specify the file name to search for
        zip_file_name = self.stock_code + ".zip"
        zip_path = os.path.join(self.root_data_dir, zip_file_name)
        extracted_dir = os.path.join(self.root_data_dir, self.stock_code)

        # Check if the extracted directory already exists
        if not os.path.exists(extracted_dir):
            # Check if the .zip file exists in the specified directory
            if os.path.exists(zip_path):
                # Unzip the file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.root_data_dir)
                print(f"Extracted {zip_file_name} to {self.root_data_dir}")
            else:
                print(f"{zip_file_name} not found in {self.root_data_dir}")
        else:
            print(f"Directory {extracted_dir} already exists, skipping extraction.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        import os
        
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_data = file.read()
            
        text_data = raw_data.split(" : ")[-1]
        time_data = raw_data.split(" : ")[0]
            
        return text_data, time_data


class Sentiment:
    """
    A class to handle sentiment analysis of financial text data using FinTwitBERT model.
    """
    
    def __init__(self, root_data_dir, stock_code, device=1):
        """
        Initialize the Sentiment analyzer.
        
        Args:
            root_data_dir (str): Root directory containing the dataset
            stock_code (str): Stock code/folder name for the dataset
            device (int): Device to use for model inference (default: 1 for GPU)
        """
        self.root_data_dir = root_data_dir
        self.stock_code = stock_code
        self.device = device
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        self.sentiment_classifier = None
        
        # Data storage
        self.dataset = None
        self.loader = None
        self.df = None
        self.df_daily = None
        
        # Results storage
        self.log_positive_score = []
        self.log_neutral_score = []
        self.log_negative_score = []
        self.log_datetime = []
        
    def setup_model(self):
        """Load and setup the FinTwitBERT sentiment model."""
        from transformers import BertForSequenceClassification, AutoTokenizer, pipeline
        
        print("Loading FinTwitBERT sentiment model...")
        self.model = BertForSequenceClassification.from_pretrained(
            "StephanAkkerman/FinTwitBERT-sentiment",
            num_labels=3,
            id2label={0: "neutral", 1: "positive", 2: "negative"},
            label2id={"neutral": 0, "positive": 1, "negative": 2},
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "StephanAkkerman/FinTwitBERT-sentiment"
        )

        # Construct Huggingface pipeline
        self.sentiment_classifier = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=self.device, 
            top_k=None, 
            padding=True, 
            truncation=True, 
            max_length=256
        )
        print("Model setup complete!")
        
    def setup_dataset(self, batch_size=32):
        """Setup the dataset and dataloader."""
        from torch.utils.data import DataLoader
        
        print(f"Setting up dataset from {self.root_data_dir}/{self.stock_code}")
        self.dataset = MyTextDataset(self.root_data_dir, self.stock_code)
        self.loader = DataLoader(self.dataset, batch_size=batch_size)
        print(f"Dataset ready with {len(self.dataset)} samples")
        
    def predict_sentiment(self):
        """Run sentiment prediction on all data."""
        import torch
        from tqdm import tqdm
        
        if self.sentiment_classifier is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        if self.loader is None:
            raise ValueError("Dataset not setup. Call setup_dataset() first.")
            
        print("Running sentiment prediction...")
        
        # Reset logs
        self.log_positive_score = []
        self.log_neutral_score = []
        self.log_negative_score = []
        self.log_datetime = []

        for (input_data, time_data) in tqdm(self.loader):
            with torch.cuda.amp.autocast():
                outputs = self.sentiment_classifier(list(input_data), batch_size=len(input_data))

            for (output, td) in zip(outputs, time_data):
                for label in output:
                    if label["label"] == 'positive':
                        self.log_positive_score.append(label["score"])
                    elif label["label"] == 'neutral':
                        self.log_neutral_score.append(label["score"])
                    elif label["label"] == 'negative':
                        self.log_negative_score.append(label["score"])

                self.log_datetime.append(td)
                
        print(f"Sentiment prediction complete! Processed {len(self.log_datetime)} samples")
        
    def organize_data(self):
        """Organize sentiment data into pandas DataFrame."""
        import pandas as pd
        
        if not self.log_datetime:
            raise ValueError("No sentiment data found. Run predict_sentiment() first.")
            
        print("Organizing data into DataFrame...")
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'date': self.log_datetime,
            'positive': self.log_positive_score,
            'neutral': self.log_neutral_score,
            'negative': self.log_negative_score,
        })
        
        # Set index to datetime
        self.df['date'] = pd.to_datetime(self.df['date'], utc=True)
        self.df.set_index('date', inplace=True)
        self.df.sort_index(inplace=True)
        
        print(f"Data organized! DataFrame shape: {self.df.shape}")

    def calculate_sentiment_scores(self, window=28):
        """Calculate sentiment scores and daily aggregations."""
        if self.df is None:
            raise ValueError("Data not organized. Call organize_data() first.")
            
        print("Calculating sentiment scores...")
        
        # Calculate composite sentiment score
        self.df['score'] = (self.df['positive'] - self.df['negative']) / (self.df['positive'] + self.df['negative'])

        # Calculate daily median sentiment
        self.df_daily = self.df.resample('D').median()
        self.df_daily.fillna(0, inplace=True)

        # Count entries per day
        df_daily_count = self.df.resample('D').size()
        self.df_daily['entry_count'] = df_daily_count
        self.df_daily['entry_count_ma'] = self.df_daily['entry_count'].rolling(window=window, center=False).mean()

        # Calculate moving average of sentiment score
        self.df_daily['score_ma'] = self.df_daily['score'].rolling(window=window, center=False).mean()
        
        print(f"Sentiment scores calculated! Daily data shape: {self.df_daily.shape}")
        
    def plot_sentiment(self, figsize=(12, 6)):
        """Plot sentiment scores over time."""
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        if self.df_daily is None:
            raise ValueError("Daily data not calculated. Call calculate_sentiment_scores() first.")
            
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot sentiment scores
        ax1.plot(self.df_daily.index, 
                 self.df_daily['score'], color='blue', label='Sentiment Score')
        ax1.plot(self.df_daily.index, 
                 self.df_daily['score_ma'], color='red', label='Sentiment Score 28d MA')

        ax1.set_xlabel('Date', fontsize=16)
        ax1.set_ylabel('Sentiment', color='blue', fontsize=16)
        ax1.tick_params(axis='y', labelcolor='blue')

        # Format x-axis
        ax1.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax1.xaxis.set_major_formatter(DateFormatter('%m/%Y'))

        ax1.legend(loc='upper left', fontsize=12)
        plt.title(f'Sentiment Scores {self.stock_code.upper()}', fontsize=16)
        plt.show()
        
    def run_full_analysis(self, batch_size=32, window=28):
        # """Run the complete sentiment analysis pipeline."""
        # print("=" * 50)
        # print("Starting Full Sentiment Analysis Pipeline")
        # print("=" * 50)
        
        self.setup_model()
        self.setup_dataset(batch_size=batch_size)
        self.predict_sentiment()
        self.organize_data()
        self.calculate_sentiment_scores(window=window)
        
        # print("=" * 50)
        # print("Sentiment Analysis Complete!")
        # print("=" * 50)
        # print(f"Total samples processed: {len(self.df)}")
        # print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        # print(f"Average daily sentiment: {self.df_daily['score'].mean():.3f}")
        
        return self.df, self.df_daily
        
    def get_summary_stats(self):
        """Get summary statistics of the sentiment analysis."""
        if self.df_daily is None:
            raise ValueError("Analysis not complete. Run run_full_analysis() first.")
            
        stats = {
            'total_samples': len(self.df),
            'date_range': (self.df.index.min(), self.df.index.max()),
            'avg_daily_sentiment': self.df_daily['score'].mean(),
            'sentiment_std': self.df_daily['score'].std(),
            'avg_daily_comments': self.df_daily['entry_count'].mean(),
            'total_days': len(self.df_daily),
            'positive_days': len(self.df_daily[self.df_daily['score'] > 0]),
            'negative_days': len(self.df_daily[self.df_daily['score'] < 0]),
            'neutral_days': len(self.df_daily[self.df_daily['score'] == 0])
        }
        
        return stats