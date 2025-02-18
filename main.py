import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json


class ArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.pending_bits = 0
        self.output_bits = []

    def output_bit(self, bit):
        self.output_bits.append(bit)
        while self.pending_bits > 0:
            self.output_bits.append(1 - bit)
            self.pending_bits -= 1

    def encode_symbol(self, cum_freq, freq, total):
        range_val = self.high - self.low + 1
        self.high = self.low + (range_val * (cum_freq + freq)) // total - 1
        self.low = self.low + (range_val * cum_freq) // total
        while True:
            if self.high < self.half_range:
                self.output_bit(0)
                self.low = self.low * 2
                self.high = self.high * 2 + 1
            elif self.low >= self.half_range:
                self.output_bit(1)
                self.low = (self.low - self.half_range) * 2
                self.high = (self.high - self.half_range) * 2 + 1
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.pending_bits += 1
                self.low = (self.low - self.quarter_range) * 2
                self.high = (self.high - self.quarter_range) * 2 + 1
            else:
                break

    def finish(self):
        self.pending_bits += 1
        if self.low < self.quarter_range:
            self.output_bit(0)
        else:
            self.output_bit(1)
        while len(self.output_bits) % 8 != 0:
            self.output_bits.append(0)
        b = bytearray()
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.output_bits[i + j]
            b.append(byte)
        return bytes(b)


class ArithmeticDecoder:
    def __init__(self, bitstream, precision=32):
        self.precision = precision
        self.full_range = 1 << precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.full_range >> 2
        self.low = 0
        self.high = self.full_range - 1
        self.code = 0
        self.bitstream = self._bits_from_bytes(bitstream)
        self.bit_index = 0
        for _ in range(precision):
            self.code = (self.code << 1) | self._read_bit()

    def _bits_from_bytes(self, b):
        bits = []
        for byte in b:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def _read_bit(self):
        if self.bit_index < len(self.bitstream):
            bit = self.bitstream[self.bit_index]
            self.bit_index += 1
            return bit
        return 0

    def decode_target(self, total):
        range_val = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // range_val

    def update(self, cum_freq, freq, total):
        range_val = self.high - self.low + 1
        self.high = self.low + (range_val * (cum_freq + freq)) // total - 1
        self.low = self.low + (range_val * cum_freq) // total
        while True:
            if self.high < self.half_range:
                pass
            elif self.low >= self.half_range:
                self.low -= self.half_range
                self.high -= self.half_range
                self.code -= self.half_range
            elif self.low >= self.quarter_range and self.high < 3 * self.quarter_range:
                self.low -= self.quarter_range
                self.high -= self.quarter_range
                self.code -= self.quarter_range
            else:
                break
            self.low *= 2
            self.high = self.high * 2 + 1
            self.code = self.code * 2 + self._read_bit()

# Frequency Calculation
def get_frequencies(probs, scale=20000):
    freqs = [max(1, int(p * scale)) for p in probs]
    total = sum(freqs)
    cum_freq = [0]
    for f in freqs:
        cum_freq.append(cum_freq[-1] + f)
    return freqs, cum_freq, total

# Compress Text
def compress_text(text, model, tokenizer, scale=100000, device="cpu"):
    model.eval()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    encoder = ArithmeticEncoder(precision=32)
    context_ids = []

    with torch.no_grad():
        for token in tokens:
            input_ids = torch.tensor([context_ids[-1024:] if len(context_ids) > 1024 else context_ids], dtype=torch.long).to(device)
            if input_ids.numel() == 0:
                input_ids = torch.tensor([[tokenizer.bos_token_id or 0]], dtype=torch.long).to(device)
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :].to("cpu").numpy()
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            freqs, cum_freq, total = get_frequencies(probs, scale=scale)
            encoder.encode_symbol(cum_freq[token], freqs[token], total)
            context_ids.append(token)

    bitstream = encoder.finish()
    return bitstream, tokens

# Decompress Text
def decompress_text(bitstream, token_count, model, tokenizer, scale=100000, device="cpu"):
    model.eval()
    decoded_tokens = []
    decoder = ArithmeticDecoder(bitstream, precision=32)
    context_ids = []

    with torch.no_grad():
        for _ in range(token_count):
            input_ids = torch.tensor([context_ids[-1024:] if len(context_ids) > 1024 else context_ids], dtype=torch.long).to(device)
            if input_ids.numel() == 0:
                input_ids = torch.tensor([[tokenizer.bos_token_id or 0]], dtype=torch.long).to(device)
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :].to("cpu").numpy()
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            freqs, cum_freq, total = get_frequencies(probs, scale=scale)
            target = decoder.decode_target(total)
            symbol = None

            for s in range(len(freqs)):
                if cum_freq[s] <= target < cum_freq[s + 1]:
                    symbol = s
                    decoder.update(cum_freq[s], freqs[s], total)
                    break

            if symbol is None:
                raise ValueError("Decoding failed; symbol not found.")

            decoded_tokens.append(symbol)
            context_ids.append(symbol)

    decoded_text = tokenizer.decode(decoded_tokens)
    return decoded_text

def main():
    input_filename = "input.txt"
    if not os.path.exists(input_filename):
        print(f"Please create a file named '{input_filename}' with some text to compress.")
        return

    with open(input_filename, "r", encoding="utf-8") as f:
        text = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    print("Compressing text...")
    bitstream, tokens = compress_text(text, model, tokenizer, scale=1000000, device=device)
    print(f"Compressed bitstream length: {len(bitstream)} bytes")
    with open("compressed.bin", "wb") as f:
        f.write(bitstream)
    metadata = {"token_count": len(tokens)}
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    print("Compression complete. Files 'compressed.bin' and 'metadata.json' written.")

    with open("compressed.bin", "rb") as f:
        bitstream = f.read()
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    token_count = metadata["token_count"]

    print("Decompressing text...")
    decoded_text = decompress_text(bitstream, token_count, model, tokenizer, scale=1000000, device=device)
    print("Decompression complete.")
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(decoded_text)
    print("Decompressed text saved to 'output.txt'.")

if __name__ == "__main__":
    main()