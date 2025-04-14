import heapq  
import os
import numpy as np   

class HuffmanNode:  
    def __init__(self, pixel, freq):  
        self.pixel = pixel  
        self.freq = freq  
        self.left = None  
        self.right = None  

    def __lt__(self, other):  
        return self.freq < other.freq  

def build_huffman_tree(freq_dict):  
    heap = [HuffmanNode(pixel, freq) for pixel, freq in freq_dict.items()]  
    heapq.heapify(heap)  
    while len(heap) > 1:  
        left = heapq.heappop(heap)  
        right = heapq.heappop(heap)  
        merged = HuffmanNode(None, left.freq + right.freq)  
        merged.left = left  
        merged.right = right  
        heapq.heappush(heap, merged)  
    return heap[0]  

def generate_codes(node, current_code, codes):  
    if node.pixel is not None:  
        codes[node.pixel] = current_code  
        return  
    generate_codes(node.left, current_code + "0", codes)  
    generate_codes(node.right, current_code + "1", codes)  

def huffman_encode(img):  
    freq = {}  
    for pixel in img.flatten():  
        freq[pixel] = freq.get(pixel, 0) + 1  
    root = build_huffman_tree(freq)  
    codes = {}  
    generate_codes(root, "", codes)  
    encoded_bits = ''.join([codes[pixel] for pixel in img.flatten()])  
    return encoded_bits, codes  

def huffman_decode(encoded_bits, codes, shape):  
    reverse_codes = {v: k for k, v in codes.items()}  
    current_code = ""  
    decoded_pixels = []  
    for bit in encoded_bits:  
        current_code += bit  
        if current_code in reverse_codes:  
            decoded_pixels.append(reverse_codes[current_code])  
            current_code = ""  
    return np.array(decoded_pixels).reshape(shape)  