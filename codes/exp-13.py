import re
from collections import Counter
from google.colab import files

# 1. Prompt the user to upload their text file
print("Please upload your text file (e.g., 'sample_text.txt' or any other document):")
uploaded_text = files.upload()

# 2. Process the uploaded file
for filename in uploaded_text.keys():
    # Read the text content
    content = uploaded_text[filename].decode('utf-8', errors='ignore')
    print(f"\nSuccessfully loaded '{filename}'!")
    
    # Replace non-alphanumeric characters with spaces to separate concatenated columns/words
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', ' ', content).lower()
    
    # Split into words on any whitespace
    words = cleaned_text.split()
    
    # Filter out purely numeric values (like years or IDs) if analyzing word frequencies
    words = [word for word in words if not word.isdigit()]
    
    # Calculate frequency distribution
    word_counts = Counter(words)
    
    # Display the top 15 most common words
    print("\nWord Frequency Distribution (Top 15):")
    print("-" * 40)
    for word, count in word_counts.most_common(15):
        print(f"{word:<20}: {count}")
