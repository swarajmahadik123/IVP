import pandas as pd  
import os
import cv2
import numpy as np
from utils import *
from rle_compression import *
from huffman_compression import *
import pickle

def test_compression(image_path):
    img = load_image(image_path)
    filename = os.path.basename(image_path)
    
    # Create directories if they don't exist
    os.makedirs("data/compressed", exist_ok=True)
    os.makedirs("data/decompressed", exist_ok=True)
    
    # RLE Compression
    rle_encoded = rle_encode(img)
    with open(f"data/compressed/rle_{filename}.txt", "w") as f:
        f.write(str(rle_encoded))
    
    # RLE Decompression
    rle_decoded = rle_decode(rle_encoded, img.shape)
    save_image(rle_decoded, f"data/decompressed/rle_{filename}")
    
    # Huffman Compression
    huffman_bits, huffman_codes = huffman_encode(img)
    with open(f"data/compressed/huffman_{filename}.txt", "w") as f:
        f.write(huffman_bits)
    with open(f"data/compressed/huffman_{filename}_codes.pkl", "wb") as f:
        pickle.dump(huffman_codes, f)
    
    # Huffman Decompression
    huffman_decoded = huffman_decode(huffman_bits, huffman_codes, img.shape)
    save_image(huffman_decoded, f"data/decompressed/huffman_{filename}")
    
    # Calculate and return metrics
    mse_rle, psnr_rle = calculate_metrics(img, rle_decoded)
    mse_huffman, psnr_huffman = calculate_metrics(img, huffman_decoded)
    
    return {
        "Image": filename,
        "RLE_Compression_Ratio": os.path.getsize(image_path)/len(str(rle_encoded)),
        "Huffman_Compression_Ratio": os.path.getsize(image_path)/(len(huffman_bits)/8),
        "RLE_PSNR": psnr_rle,
        "Huffman_PSNR": psnr_huffman
    }
    
if __name__ == "__main__":  
    results = []  
    for img_file in os.listdir("data/raw_images"):  
        if img_file.endswith((".png", ".jpg", ".dcm")):  
            result = test_compression(f"data/raw_images/{img_file}")  
            results.append(result)  
    # Save results to CSV  
    pd.DataFrame(results).to_csv("results/compression_ratios.csv", index=False)  