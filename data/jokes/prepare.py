import os
import pandas as pd
import tiktoken
import numpy as np

# Set file paths
csv_file_path = os.path.join(os.path.dirname(__file__), 'jokes.csv')

# Read the jokes CSV file
df = pd.read_csv(csv_file_path)

# Combine all jokes into one string (concatenating them)
data = '\n'.join(df['Joke'].tolist())

# Split into training (90%) and validation (10%) data
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Encode the text using GPT-2 BPE encoding
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

# Print the number of tokens in each set
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin and val.bin will now contain the tokenized jokes
