import heapq
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
    if len(img.shape) == 3:  # Color image
        h, w, c = img.shape
        # Treat each pixel (including all channels) as a single symbol
        pixels = [tuple(pixel) for pixel in img.reshape(-1, c)]
    else:  # Grayscale
        pixels = img.flatten()
    
    freq = {}
    for pixel in pixels:
        freq[pixel] = freq.get(pixel, 0) + 1
    
    root = build_huffman_tree(freq)
    codes = {}
    generate_codes(root, "", codes)
    encoded_bits = ''.join([codes[pixel] for pixel in pixels])
    return encoded_bits, codes, img.shape

def huffman_decode(encoded_bits, codes, shape):
    reverse_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_pixels = []
    
    for bit in encoded_bits:
        current_code += bit
        if current_code in reverse_codes:
            decoded_pixels.append(reverse_codes[current_code])
            current_code = ""
    
    if len(shape) == 3:  # Color image
        return np.array(decoded_pixels).reshape(shape)
    else:  # Grayscale
        return np.array(decoded_pixels).reshape(shape)

def rle_encode(img):
    # Convert color image to a 2D array (height x (width*channels))
    if len(img.shape) == 3:  # Color image
        h, w, c = img.shape
        flat_img = img.reshape(-1, c)
    else:  # Grayscale
        flat_img = img.flatten()
        c = 1
    
    encoded = []
    count = 1
    for i in range(1, len(flat_img)):
        if np.array_equal(flat_img[i], flat_img[i-1]):
            count += 1
        else:
            encoded.append((tuple(flat_img[i-1]) if c > 1 else flat_img[i-1], count))
            count = 1
    encoded.append((tuple(flat_img[-1]) if c > 1 else flat_img[-1], count))
    return encoded, img.shape

def rle_decode(encoded, shape):
    try:
        decoded_pixels = []
        
        for val, count in encoded:
            # Handle both grayscale and color cases
            if isinstance(val, tuple):  # Color
                decoded_pixels.extend([list(val)] * count)
            else:  # Grayscale
                decoded_pixels.extend([val] * count)
        
        # Convert to numpy array and reshape
        if len(shape) == 3:  # Color image
            pixel_array = np.array(decoded_pixels, dtype=np.uint8)
            return pixel_array.reshape(shape)
        else:  # Grayscale
            return np.array(decoded_pixels, dtype=np.uint8).reshape(shape)
            
    except Exception as e:
        print(f"Error in rle_decode: {str(e)}")
        raise
    
    
def huffman_decode(encoded_bits, codes, shape):
    try:
        # Create reverse mapping from code to pixel
        reverse_codes = {v: k for k, v in codes.items()}
        current_code = ""
        decoded_pixels = []

        # Convert bytes to bitstring if needed
        if isinstance(encoded_bits, bytes):
            bitstring = ''.join(f'{byte:08b}' for byte in encoded_bits)
        else:
            bitstring = encoded_bits

        for bit in bitstring:
            current_code += bit
            if current_code in reverse_codes:
                pixel = reverse_codes[current_code]
                decoded_pixels.append(pixel)
                current_code = ""

        # Validate pixel count
        expected_pixels = shape[0] * shape[1]
        if len(decoded_pixels) != expected_pixels:
            raise ValueError(f"Expected {expected_pixels} pixels, got {len(decoded_pixels)}")

        # Convert to numpy array
        if len(shape) == 3:  # Color image
            return np.array(decoded_pixels, dtype=np.uint8).reshape(shape)
        else:  # Grayscale
            return np.array(decoded_pixels, dtype=np.uint8).reshape(shape)

    except Exception as e:
        raise RuntimeError(f"Huffman decoding failed: {str(e)}") from e
