import io
import string
from collections import Counter
import pandas as pd
from google.colab import files

# Step 1: Upload the customer reviews CSV file
print("Please upload your customer reviews CSV file:")
uploaded = files.upload()

if uploaded:
    # Get the file name
    file_name = list(uploaded.keys())[0]
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))
    
    print(f"\nSuccessfully loaded '{file_name}' with shape: {df.shape}")
    print("\nColumns in dataset:", list(df.columns))
    
    # Step 2: Dynamically detect the text/review column
    review_col = None
    # Check for common names like 'review', 'text', 'comment', 'content'
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['review', 'text', 'comment', 'content', 'body']):
            review_col = col
            break
            
    # If no keyword matches, select the first object/string column
    if not review_col:
        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            review_col = string_cols[0]
            
    if review_col:
        print(f"\nAnalyzing text in column: '{review_col}'")
        
        # Drop null values and combine all reviews into a single text block
        all_reviews = df[review_col].dropna().astype(str)
        
        # Step 3: Tokenize, clean, and count word frequencies
        word_counter = Counter()
        
        for review in all_reviews:
            # Convert to lowercase and remove punctuation
            cleaned_review = review.lower().translate(str.maketrans('', '', string.punctuation))
            # Split into individual words
            words = cleaned_review.split()
            word_counter.update(words)
            
        # Step 4: Convert to DataFrame for presentation
        freq_df = pd.DataFrame(word_counter.most_common(), columns=['Word', 'Frequency'])
        
        print(f"\nTotal unique words found: {len(freq_df)}")
        print("\nTop 30 Most Frequent Words:")
        print(freq_df.head(30).to_string(index=False))
    else:
        print("\nError: Could not find a text or review column in the uploaded dataset.")
else:
    print("No file was uploaded.")
  
