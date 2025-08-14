# Import required libraries
import praw
import os
from datetime import datetime
from pytz import timezone
import time
import glob
from collections import defaultdict

class RedditScraper:
    """
    A class to handle Reddit data collection and storage for sentiment analysis.
    """
    
    def __init__(self, client_id, client_secret, user_agent, username=None, password=None):
        """
        Initialize the Reddit scraper.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for Reddit API
            username (str, optional): Reddit username for authenticated requests
            password (str, optional): Reddit password for authenticated requests
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.username = username
        self.password = password
        
        # Initialize Reddit client
        self.reddit = None
        self.setup_reddit_client()
        
    def setup_reddit_client(self):
        """Initialize the Reddit API client."""
        try:
            if self.username and self.password:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    username=self.username,
                    password=self.password
                )
            else:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
            
            print("✅ Reddit API client initialized successfully!")
            print(f"Read-only mode: {self.reddit.read_only}")
            
            # Test connection
            test_sub = self.reddit.subreddit('test')
            print(f"User: {self.reddit.user.me() if not self.reddit.read_only else 'Read-only mode'}")
            
        except Exception as e:
            print(f"❌ Error initializing Reddit API: {e}")
            self.reddit = None
            
    def save_submission_to_files(self, submission, output_dir="reddit_data"):
        """
        Save a Reddit submission and its comments to individual text files by timestamp.
        
        Args:
            submission: Reddit submission object
            output_dir (str): Directory to save files
            
        Returns:
            tuple: (list of saved file paths, comment count) or None if error
        """
        try:
            import os
            from datetime import datetime
            from pytz import timezone
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Dictionary to collect texts by minute-level timestamp
            timestamp_data = {}
            
            # Process submission itself
            utc_dt = datetime.fromtimestamp(submission.created_utc, tz=timezone('UTC'))
            aest_dt = utc_dt.astimezone(timezone('Australia/Sydney'))
            
            # Create minute-level timestamp key (ignore seconds)
            minute_key = aest_dt.strftime('%Y_%m_%d_%H_%M')
            iso_timestamp = aest_dt.strftime('%Y-%m-%dT%H:%M:%S%z')
            iso_timestamp = iso_timestamp[:-2] + ':' + iso_timestamp[-2:]
            
            # Initialize with submission
            if minute_key not in timestamp_data:
                timestamp_data[minute_key] = []
            
            submission_text = submission.title
            if submission.selftext:
                submission_text += f" {submission.selftext}"
            
            timestamp_data[minute_key].append({
                'timestamp': iso_timestamp,
                'text': submission_text,
                'type': 'submission'
            })
            
            # Process all comments
            submission.comments.replace_more(limit=None)
            comment_count = 0
            
            for comment in submission.comments.list():
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    # Convert comment timestamp to AEST
                    comment_utc = datetime.fromtimestamp(comment.created_utc, tz=timezone('UTC'))
                    comment_aest = comment_utc.astimezone(timezone('Australia/Sydney'))
                    
                    # Create minute-level timestamp key
                    comment_minute_key = comment_aest.strftime('%Y_%m_%d_%H_%M')
                    comment_iso = comment_aest.strftime('%Y-%m-%dT%H:%M:%S%z')
                    comment_iso = comment_iso[:-2] + ':' + comment_iso[-2:]
                    
                    if comment_minute_key not in timestamp_data:
                        timestamp_data[comment_minute_key] = []
                    
                    timestamp_data[comment_minute_key].append({
                        'timestamp': comment_iso,
                        'text': comment.body,
                        'type': 'comment'
                    })
                    comment_count += 1
            
            # Save each minute's data to separate files
            saved_files = []
            for minute_key, entries in timestamp_data.items():
                filename = f"{minute_key}.txt"
                filepath = os.path.join(output_dir, filename)
                
                # Sort entries by exact timestamp within the minute
                entries.sort(key=lambda x: x['timestamp'])
                
                content_lines = []
                for entry in entries:
                    content_lines.append(f"{entry['timestamp']} : {entry['text']}")
                
                content = '\n'.join(content_lines)
                
                # Save to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                saved_files.append(filepath)
            
            return saved_files, comment_count
            
        except Exception as e:
            print(f"Error saving submission to files: {e}")
            return None
            
    def scrape_subreddit(self, subreddit_name, search_term, limit=1000, output_dir="reddit_data", 
                        datasets_dir="./datasets", delay=3):
        """
        Scrape a subreddit for posts containing a search term and save to files.
        
        Args:
            subreddit_name (str): Name of the subreddit to search
            search_term (str): Term to search for
            limit (int): Maximum number of submissions to process
            output_dir (str): Directory to save individual files
            datasets_dir (str): Directory to save the final zip file
            delay (int): Delay between API calls in seconds
            
        Returns:
            tuple: (list of all saved files, zip file path)
        """
        if not self.reddit:
            print("❌ Reddit client not initialized. Cannot scrape data.")
            return None, None
            
        import os
        import zipfile
        import shutil
        import time
        
        saved_files = []
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(datasets_dir, exist_ok=True)
        
        print(f"🔍 Searching r/{subreddit_name} for '{search_term}'...")
        print(f"📁 Saving files to '{output_dir}' directory...")
        
        try:
            submissions = self.reddit.subreddit(subreddit_name).search(search_term, limit=limit)
            
            for i, submission in enumerate(submissions):
                print(f"\n📄 Processing submission {i+1}: '{submission.title[:50]}...'")
                print(f"   💬 Comments: {submission.num_comments}")
                
                file_results = self.save_submission_to_files(submission, output_dir)
                if file_results:
                    files_saved, comment_count = file_results
                    saved_files.extend(files_saved)
                    print(f"   ✅ Created {len(files_saved)} files ({comment_count} comments processed)")
                    
                    # Show first few files created
                    for filepath in files_saved[:3]:
                        print(f"      - {os.path.basename(filepath)}")
                    if len(files_saved) > 3:
                        print(f"      ... and {len(files_saved) - 3} more files")
                
                # Add delay to avoid overwhelming the API
                if i > 0 and delay > 0:
                    time.sleep(delay)
            
            print(f"\n📊 Total files saved: {len(saved_files)}")
            print(f"📂 Files saved in: {os.path.abspath(output_dir)}")
            
            # Create zip file
            zip_path = None
            if saved_files:
                zip_path = self.create_zip_file(saved_files, output_dir, datasets_dir, search_term.lower())
                
            return saved_files, zip_path
            
        except Exception as e:
            print(f"❌ Error during scraping process: {e}")
            return saved_files, None
            
    def create_zip_file(self, file_list, source_dir, datasets_dir, zip_name):
        """
        Create a zip file containing all scraped data.
        
        Args:
            file_list (list): List of file paths to zip
            source_dir (str): Source directory containing the files
            datasets_dir (str): Directory to save the zip file
            zip_name (str): Name for the zip file (without extension)
            
        Returns:
            str: Path to created zip file
        """
        import os
        import zipfile
        
        zip_filename = f"{zip_name}.zip"
        zip_path = os.path.join(datasets_dir, zip_filename)
        
        print(f"\n📦 Creating zip file: {zip_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for filepath in file_list:
                    # Get relative path for zip archive
                    arcname = os.path.relpath(filepath, os.path.dirname(source_dir))
                    zipf.write(filepath, arcname)
                    
            print(f"✅ Created zip file with {len(file_list)} files: {zip_path}")
            print(f"📏 Zip file size: {os.path.getsize(zip_path):,} bytes")
            
            return zip_path
            
        except Exception as e:
            print(f"❌ Error creating zip file: {e}")
            return None
            
    def verify_zip_file(self, zip_path):
        """
        Verify and display information about a zip file.
        
        Args:
            zip_path (str): Path to the zip file to verify
        """
        import os
        import zipfile
        
        if os.path.exists(zip_path):
            print(f"✅ Zip file found: {zip_path}")
            print(f"📏 File size: {os.path.getsize(zip_path):,} bytes")
            
            # List contents of zip file
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                print(f"\n📦 Zip contains {len(file_list)} files:")
                
                # Show first few files
                for filename in file_list[:10]:
                    print(f"   - {filename}")
                
                if len(file_list) > 10:
                    print(f"   ... and {len(file_list) - 10} more files")
                    
                # Show a sample file content
                if file_list:
                    sample_file = file_list[0]
                    print(f"\n📄 Sample content from {sample_file}:")
                    with zipf.open(sample_file, 'r') as f:
                        content = f.read().decode('utf-8')
                        print(f"   '{content[:200]}{'...' if len(content) > 200 else ''}'")
                        
            print(f"\n🎯 The zip file is ready for sentiment analysis!")
            
        else:
            print(f"❌ Zip file not found at: {zip_path}")
            
    def cleanup_temp_files(self, directory):
        """
        Remove temporary files after zipping.
        
        Args:
            directory (str): Directory to remove
        """
        import shutil
        import os
        
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"🧹 Cleaned up temporary directory: {directory}")
        else:
            print(f"⚠️  Directory not found: {directory}")
            
    def get_summary_stats(self, saved_files):
        """
        Get summary statistics about the scraped data.
        
        Args:
            saved_files (list): List of saved file paths
            
        Returns:
            dict: Summary statistics
        """
        if not saved_files:
            return {}
            
        import os
        from collections import defaultdict
        
        # Group by day
        daily_activity = defaultdict(int)
        total_size = 0
        
        for filepath in saved_files:
            filename = os.path.basename(filepath)
            date_part = filename[:10]  # YYYY_MM_DD
            daily_activity[date_part] += 1
            total_size += os.path.getsize(filepath)
            
        stats = {
            'total_files': len(saved_files),
            'total_size_bytes': total_size,
            'daily_breakdown': dict(daily_activity),
            'date_range': (min(daily_activity.keys()), max(daily_activity.keys())) if daily_activity else None,
            'avg_files_per_day': len(saved_files) / len(daily_activity) if daily_activity else 0
        }
        
        return stats