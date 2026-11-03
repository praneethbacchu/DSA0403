import io
import string
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Define standard stopwords
STOPWORDS = {
    'the', 'and', 'is', 'to', 'in', 'of', 'a', 'it', 'for', 'on', 'with', 'as', 'at', 'by', 'an', 'be', 'this', 
    'that', 'from', 'but', 'are', 'was', 'your', 'my', 'have', 'has', 'had', 'we', 'they', 'you', 'he', 'she', 
    'it', 'them', 'their', 'our', 'us', 'i', 'not', 'or', 'so', 'if', 'about', 'out', 'up', 'down', 'about', 
    'will', 'would', 'could', 'should', 'can', 'also', 'said', 'there', 'their', 'which', 'who', 'how', 'what'
}

# Step 1: Load the dataset via CSV upload
print("Please upload your 'data.csv' (or any CSV file containing customer feedback):")
uploaded = files.upload()

if uploaded:
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))
    print(f"\nSuccessfully loaded '{file_name}' with shape {df.shape}.")
    
    # Determine the correct text column
    col_name = 'feedback' if 'feedback' in df.columns else None
    if not col_name:
        # Fallback to look for text-like columns if 'feedback' isn't explicitly named
        for col in df.columns:
            if any(k in col.lower() for k in ['text', 'comment', 'feedback', 'review']):
                col_name = col
                break
        if not col_name:
            col_name = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else None

    if col_name:
        print(f"Processing text from column: '{col_name}'")
        
        # Step 2: Preprocess text data (convert to lowercase, remove punctuation and stopwords)
        word_counter = Counter()
        feedback_series = df[col_name].dropna().astype(str)
        
        for text in feedback_series:
            # Remove punctuation
            cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
            # Convert to lowercase and split into words
            words = cleaned_text.lower().split()
            # Filter out stopwords and numeric items
            filtered_words = [word for word in words if word not in STOPWORDS and word.isalpha()]
            word_counter.update(filtered_words)
            
        # Convert to DataFrame for easier manipulation
        freq_df = pd.DataFrame(word_counter.most_common(), columns=['Word', 'Frequency'])
        
        if not freq_df.empty:
            # Step 3: Get user input N
            try:
                N = int(input("\nEnter the number of top frequent words to display and plot (e.g., 10, 15, 20): ") or "15")
            except ValueError:
                print("Invalid input. Defaulting to N=15.")
                N = 15
                
            top_n = freq_df.head(N)
            
            # Step 4: Display the top N most frequent words
            print(f"\nTop {N} Most Frequent Words:")
            print(top_n.to_string(index=False))
            
            # Step 5: Plot a bar graph to visualize top N words
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Frequency', y='Word', data=top_n, palette='viridis')
            plt.title(f'Top {N} Most Frequent Words in Feedback (Stopwords Removed)', fontsize=14)
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Word', fontsize=12)
            plt.tight_layout()
            plt.show()
        else:
            print("No words remaining after preprocessing.")
    else:
        print("Could not identify a text column for analysis.")
else:
    print("No file was uploaded.")
  
