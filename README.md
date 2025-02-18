### Usage
The code can be run locally or if you want to access GPU on googla colab you can copy and paste the code into a GPU runtime. It handles both running on CPU or GPU.

# Step 1: Install dependencies 
Execute on terminal:
```
pip install -r requirements.txt
```

# Step 2: Prepare Input File
Create a file named input.txt in the root directory of the project. Add the text you want to compress.

# Step 3: Run the Script
Execute the script to compress and decompress the text

```
python main.py
```

### Output Files

compressed.bin: The compressed binary file
metadata.json: Metadata containing the number of tokens in the input text.
output.txt: The decompressed text, which should match the original input.

### How it works 

# Compression
The input text is tokenized using the GPT-2 tokenizer.
For each token, the GPT-2 model generates a probability distribution over the vocabulary.
These probabilities are converted into frequency tables and used by the arithmetic encoder to compress the tokens into a binary stream.

# Decompression

The binary stream is decoded using the arithmetic decoder.
At each step, the GPT-2 model predicts the next token based on the context, and the decoder identifies the most likely token using the frequency table.

