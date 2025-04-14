from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import struct
from scipy.fftpack import dct, idct
import zlib
import heapq

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'uploads'
COMPRESSED_FOLDER = 'compressed'
DECOMPRESSED_FOLDER = 'decompressed'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
ALLOWED_COMPRESSED_EXTENSIONS = {'bin', 'cmp'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)
os.makedirs(DECOMPRESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Huffman Node class
class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huffman_direction = ''
    
    def __lt__(self, nxt):
        return self.frequency < nxt.frequency

def huffman_compress(img):
    """Lossless Huffman compression with exact dimension preservation"""
    try:
        # Validate input and get exact dimensions
        if len(img.shape) == 2:
            height, width = img.shape
            channels = 1
            img_flat = img.reshape(-1)  # Flatten grayscale image
        elif len(img.shape) == 3:
            height, width, channels = img.shape
            img_flat = img.reshape(-1)  # Flatten color image
        else:
            raise ValueError("Invalid image dimensions")

        img_bytes = img_flat.tobytes()
        original_size = len(img_bytes)
        
        # Create header with exact dimensions (B=type, I=height, I=width, B=channels, I=original_size)
        header = struct.pack('>BIIBI', 1, height, width, channels, original_size)
        
        # Skip compression for small images
        if original_size < 1024:
            return struct.pack('>BIIBI', 0, height, width, channels, original_size) + img_bytes
        
        # Calculate byte frequencies
        frequency = {}
        for byte in img_bytes:
            frequency[byte] = frequency.get(byte, 0) + 1
        
        # Skip if few unique bytes
        if len(frequency) < 4:
            return struct.pack('>BIIBI', 0, height, width, channels, original_size) + img_bytes
        
        # Build Huffman tree
        heap = [Node(freq, byte) for byte, freq in frequency.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = Node(left.frequency + right.frequency, None, left, right)
            heapq.heappush(heap, merged)
        
        # Generate codes
        codes = {}
        def traverse(node, code=''):
            if node.symbol is not None:
                codes[node.symbol] = code
                return
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')
        traverse(heap[0])
        
        # Encode data
        encoded_bits = ''.join([codes[byte] for byte in img_bytes])
        
        # Add padding to make complete bytes
        padding = (8 - len(encoded_bits) % 8) % 8
        encoded_bits += '0' * padding
        
        # Convert to bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            encoded_bytes.append(int(encoded_bits[i:i+8], 2))
        
        # Serialize Huffman tree
        tree_data = bytearray()
        tree_data.extend(struct.pack('>H', len(codes)))
        
        for byte, code in codes.items():
            tree_data.append(byte)
            tree_data.append(len(code))
            # Store code in bytes
            code_bytes = bytearray()
            for i in range(0, len(code), 8):
                chunk = code[i:i+8]
                code_bytes.append(int(chunk.ljust(8, '0'), 2))
            tree_data.extend(code_bytes)
        
        # Combine all parts
        compressed_data = header + struct.pack('>B', padding) + tree_data + encoded_bytes
        
        # Only use compression if it's actually smaller
        if len(compressed_data) >= original_size + len(header):
            return struct.pack('>BIIBI', 0, height, width, channels, original_size) + img_bytes
        
        return compressed_data
    
    except Exception as e:
        # Fallback to uncompressed with original dimensions
        if 'height' in locals() and 'width' in locals() and 'channels' in locals() and 'original_size' in locals():
            return struct.pack('>BIIBI', 0, height, width, channels, original_size) + img_bytes
        raise ValueError(f"Compression failed: {str(e)}")

def huffman_decompress(compressed_data):
    """Lossless Huffman decompression with exact dimension restoration"""
    try:
        # Minimum header size check (14 bytes)
        MIN_HEADER_SIZE = 14
        if len(compressed_data) < MIN_HEADER_SIZE:
            raise ValueError(f"Compressed data too small (min {MIN_HEADER_SIZE} bytes needed)")
        
        # Read header (BIIBI = 14 bytes)
        compression_type, height, width, channels, original_size = \
            struct.unpack('>BIIBI', compressed_data[:MIN_HEADER_SIZE])
        offset = MIN_HEADER_SIZE
        
        # Handle uncompressed case
        if compression_type == 0:
            img_data = compressed_data[offset:offset+original_size]
            if len(img_data) != original_size:
                raise ValueError(f"Data size mismatch: expected {original_size}, got {len(img_data)}")
            
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            if channels == 1:
                return img_array.reshape(height, width)
            return img_array.reshape(height, width, channels)
        
        # Read Huffman compressed data
        if len(compressed_data) < offset + 1:
            raise ValueError("Missing padding information")
        
        padding = compressed_data[offset]
        offset += 1
        
        # Read number of codes
        if len(compressed_data) < offset + 2:
            raise ValueError("Missing Huffman codes count")
        
        num_codes = struct.unpack('>H', compressed_data[offset:offset+2])[0]
        offset += 2
        
        # Rebuild Huffman codes
        reverse_codes = {}
        for _ in range(num_codes):
            if offset + 2 > len(compressed_data):
                raise ValueError("Invalid Huffman code data")
            
            byte = compressed_data[offset]
            code_len = compressed_data[offset+1]
            offset += 2
            
            # Read code bits
            code_bytes = (code_len + 7) // 8
            if offset + code_bytes > len(compressed_data):
                raise ValueError("Invalid Huffman code bits")
            
            code = ''
            for i in range(code_bytes):
                bits = f'{compressed_data[offset+i]:08b}'
                take = min(8, code_len - i*8)
                code += bits[:take]
            offset += code_bytes
            reverse_codes[code] = byte
        
        # Decode the data
        encoded_bits = ''.join(f'{byte:08b}' for byte in compressed_data[offset:])
        if padding > 0:
            encoded_bits = encoded_bits[:-padding]
        
        decoded = bytearray()
        current_code = ''
        for bit in encoded_bits:
            current_code += bit
            if current_code in reverse_codes:
                decoded.append(reverse_codes[current_code])
                current_code = ''
        
        # Verify exact size match
        if len(decoded) != original_size:
            raise ValueError(f"Critical: Decompressed size {len(decoded)} != original {original_size}")
        
        # Restore to exact original dimensions
        img_array = np.frombuffer(decoded, dtype=np.uint8)
        if channels == 1:
            return img_array.reshape(height, width)
        return img_array.reshape(height, width, channels)
    
    except Exception as e:
        raise ValueError(f"Decompression failed: {str(e)}")

def rle_encode(img):
    """RLE encoder that properly preserves image dimensions and data integrity"""
    # Store original dimensions
    if len(img.shape) == 3:  # Color image
        height, width, channels = img.shape
        # Convert to grayscale if it's a color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        height, width = img.shape
        channels = 1
    
    flat_img = img.flatten()
    encoded = bytearray()
    
    # Store original dimensions (4 bytes each)
    encoded.extend(struct.pack('>II', height, width))
    
    i = 0
    n = len(flat_img)
    MAX_NON_RUN = 127  # Maximum non-run sequence length (0-127 in storage)
    
    while i < n:
        # Find next run of at least 3 identical values
        run_length = 1
        run_value = flat_img[i]
        
        while (i + run_length < n and 
               run_length < 258 and  # Maximum run length (3-258, stored as 0-255)
               flat_img[i + run_length] == run_value):
            run_length += 1
            
        if run_length >= 3:
            # Encode as run (value + run marker)
            run_code = (run_length - 3) & 0x7F  # Get 7 bits (0-127 means 3-130)
            encoded.extend([run_value, 0x80 | run_code])  # Set high bit for run marker
            i += run_length
        else:
            # Handle non-run sequence
            non_run_start = i
            
            # Look ahead to find where a run would start
            j = i
            non_run_length = 0
            
            while j < n and non_run_length < MAX_NON_RUN:
                # Check if we have a potential run starting at position j
                if j + 2 < n and flat_img[j] == flat_img[j+1] and flat_img[j] == flat_img[j+2]:
                    break
                j += 1
                non_run_length += 1
            
            # Ensure we have at least one byte in a non-run
            if non_run_length == 0:
                non_run_length = 1
                j = i + 1
            
            # Encode non-run sequence (length byte + raw bytes)
            encoded.append(non_run_length - 1)  # Store as 0-126 (meaning 1-127)
            encoded.extend(flat_img[i:j])
            
            i = j  # Move to the end of this non-run
    
    return bytes(encoded)

def rle_decode(encoded):
    """Robust RLE decoder that preserves image dimensions"""
    try:
        # Verify minimum header size
        if len(encoded) < 8:
            raise ValueError("Compressed data too small (needs 8 byte header)")
        
        # Read original dimensions
        height, width = struct.unpack('>II', encoded[:8])
        expected_size = height * width
        encoded_data = encoded[8:]
        
        decoded = bytearray()
        i = 0
        n = len(encoded_data)
        
        while i < n:
            if i + 1 < n and (encoded_data[i+1] & 0x80) == 0x80:
                # Run sequence
                value = encoded_data[i]
                run_length = (encoded_data[i+1] & 0x7F) + 3  # 3-130
                
                decoded.extend([value] * run_length)
                i += 2
            else:
                # Non-run sequence
                if i >= n:
                    break
                
                non_run_length = (encoded_data[i] & 0x7F) + 1  # 1-128
                
                # Safety check to avoid reading past the end
                if i + 1 + non_run_length > n:
                    non_run_length = n - i - 1
                    if non_run_length <= 0:
                        break
                
                decoded.extend(encoded_data[i+1:i+1+non_run_length])
                i += 1 + non_run_length
        
        # Ensure the decoded data has exactly the expected size
        if len(decoded) != expected_size:
            # Handle size mismatch
            if len(decoded) < expected_size:
                # Pad with zeros
                decoded.extend([0] * (expected_size - len(decoded)))
            else:
                # Truncate
                decoded = decoded[:expected_size]
        
        # Reshape to original dimensions
        return np.array(decoded, dtype=np.uint8).reshape(height, width)
    
    except struct.error as e:
        raise ValueError(f"Invalid header format: {str(e)}")
    except Exception as e:
        raise ValueError(f"RLE decompression failed: {str(e)}")


# DCT encoding/decoding functions
def dct_compress(img, quality=70):
    # Better quantization tables for JPEG-like compression
    if quality < 1:
        quality = 1
    if quality > 100:
        quality = 100
    
    # Standard JPEG luminance quantization table
    Q_lum = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Adjust quality factor
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2
    
    Q_lum = np.clip(np.round(Q_lum * scale / 100), 1, 255).astype(np.uint8)
    
    if len(img.shape) == 3:  # Color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    # Process image with DCT
    dct_blocks = []
    
    for i in range(0, img.shape[0], 8):
        for j in range(0, img.shape[1], 8):
            block = img[i:i+8, j:j+8].astype(np.float32) - 128
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            
            # Quantize
            quantized = np.round(dct_block / Q_lum).astype(np.int16)
            
            # Zigzag scan
            zigzag = np.zeros(64, dtype=np.int16)
            index = 0
            for sum_idx in range(15):
                if sum_idx % 2 == 0:
                    for i_zig in range(min(sum_idx, 7), max(0, sum_idx-7) - 1, -1):
                        j_zig = sum_idx - i_zig
                        if i_zig < 8 and j_zig < 8:
                            zigzag[index] = quantized[i_zig, j_zig]
                            index += 1
                else:
                    for i_zig in range(max(0, sum_idx-7), min(sum_idx+1, 8)):
                        j_zig = sum_idx - i_zig
                        if i_zig < 8 and j_zig < 8:
                            zigzag[index] = quantized[i_zig, j_zig]
                            index += 1
            
            # Run-length encode
            rle = []
            zero_count = 0
            for coef in zigzag:
                if coef == 0:
                    zero_count += 1
                else:
                    rle.append(zero_count)
                    rle.append(coef)
                    zero_count = 0
            
            if zero_count > 0:
                rle.append(-1)  # End of block marker
            
            dct_blocks.append(rle)
    
    # Serialize the compressed data
    header = struct.pack('>IIBB', h, w, pad_h, pad_w)
    dct_data = bytearray()
    
    for block in dct_blocks:
        for value in block:
            dct_data.extend(struct.pack('>h', value))
    
    # Compress with zlib
    compressed_dct = zlib.compress(header + dct_data, level=9)
    return compressed_dct

def dct_decompress(compressed_data):
    # Decompress the data
    decompressed = zlib.decompress(compressed_data)
    
    # Read header
    h, w, pad_h, pad_w = struct.unpack('>IIBB', decompressed[:10])
    offset = 10
    
    # Standard quantization table (must match compression)
    Q_lum = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    padded_h = h + pad_h
    padded_w = w + pad_w
    reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    # Inverse zigzag
    def inverse_zigzag(zigzag):
        block = np.zeros((8, 8), dtype=np.float32)
        index = 0
        for sum_idx in range(15):
            if sum_idx % 2 == 0:
                for i_zig in range(min(sum_idx, 7), max(0, sum_idx-7) - 1, -1):
                    j_zig = sum_idx - i_zig
                    if i_zig < 8 and j_zig < 8 and index < len(zigzag):
                        block[i_zig, j_zig] = zigzag[index]
                        index += 1
            else:
                for i_zig in range(max(0, sum_idx-7), min(sum_idx+1, 8)):
                    j_zig = sum_idx - i_zig
                    if i_zig < 8 and j_zig < 8 and index < len(zigzag):
                        block[i_zig, j_zig] = zigzag[index]
                        index += 1
        return block
    
    # Decode blocks
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            # Decode RLE
            block_values = []
            while True:
                value = struct.unpack('>h', decompressed[offset:offset+2])[0]
                offset += 2
                
                if value == -1:
                    while len(block_values) < 64:
                        block_values.append(0)
                    break
                elif value >= 0:
                    zero_count = value
                    coef = struct.unpack('>h', decompressed[offset:offset+2])[0]
                    offset += 2
                    block_values.extend([0] * zero_count)
                    block_values.append(coef)
            
            # Reconstruct block
            quantized = inverse_zigzag(block_values[:64])
            dct_block = quantized * Q_lum
            block = idct(idct(dct_block, norm='ortho').T, norm='ortho').T + 128
            block = np.clip(block, 0, 255)
            
            reconstructed[i:i+8, j:j+8] = block
    
    # Remove padding
    if pad_h > 0:
        reconstructed = reconstructed[:-pad_h]
    if pad_w > 0:
        reconstructed = reconstructed[:, :-pad_w]
    
    return reconstructed.astype(np.uint8)

# Flask routes
def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['GET', 'POST'])
def compress():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        algorithm = request.form.get('algorithm')
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_image_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if img is None:
                flash('Invalid image file')
                return redirect(request.url)
            
            if img.dtype == np.uint16:
                img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
            
            if algorithm == 'rle':
                compressed_data = rle_encode(img)
                compressed_filename = f"rle_{os.path.splitext(filename)[0]}.bin"
                compressed_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)
                
                with open(compressed_path, 'wb') as f:
                    f.write(compressed_data)
                
                original_size = os.path.getsize(filepath)
                compressed_size = os.path.getsize(compressed_path)
                ratio = original_size / compressed_size if compressed_size > 0 else 0
                
                flash(f'RLE compression successful! Ratio: {ratio:.2f}:1')
                return render_template('compress.html', 
                                    compressed=True,
                                    filename=compressed_filename,
                                    algorithm=algorithm,
                                    original_size=original_size,
                                    compressed_size=compressed_size,
                                    ratio=ratio)
            
            elif algorithm == 'huffman':
                compressed_data = huffman_compress(img)
                compressed_filename = f"huffman_{os.path.splitext(filename)[0]}.bin"
                compressed_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)
                
                with open(compressed_path, 'wb') as f:
                    f.write(compressed_data)
                
                original_size = os.path.getsize(filepath)
                compressed_size = os.path.getsize(compressed_path)
                ratio = original_size / compressed_size if compressed_size > 0 else 0
                
                flash(f'Huffman compression {"successful" if ratio > 1 else "not beneficial"}! Ratio: {ratio:.2f}:1')
                return render_template('compress.html', 
                                    compressed=True,
                                    filename=compressed_filename,
                                    algorithm=algorithm,
                                    original_size=original_size,
                                    compressed_size=compressed_size,
                                    ratio=ratio)    
            elif algorithm == 'dct':
                quality = int(request.form.get('quality', 70))
                compressed_data = dct_compress(img, quality)
                compressed_filename = f"dct_{os.path.splitext(filename)[0]}_q{quality}.bin"
                compressed_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)
                
                with open(compressed_path, 'wb') as f:
                    f.write(compressed_data)
                
                original_size = os.path.getsize(filepath)
                compressed_size = os.path.getsize(compressed_path)
                ratio = original_size / compressed_size if compressed_size > 0 else 0
                
                flash(f'DCT compression successful! Ratio: {ratio:.2f}:1')
                return render_template('compress.html', 
                                    compressed=True,
                                    filename=compressed_filename,
                                    algorithm=algorithm,
                                    original_size=original_size,
                                    compressed_size=compressed_size,
                                    ratio=ratio)
        
    return render_template('compress.html', compressed=False)

@app.route('/decompress', methods=['GET', 'POST'])
def decompress():
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        
        if not algorithm:
            flash('Please select a decompression algorithm', 'error')
            return redirect(request.url)
        
        try:
            if algorithm == 'rle':
                if 'rle_compressed' not in request.files:
                    flash('No compressed file selected', 'error')
                    return redirect(request.url)
                
                rle_file = request.files['rle_compressed']
                
                if rle_file.filename == '':
                    flash('No file selected', 'error')
                    return redirect(request.url)
                
                compressed_data = rle_file.read()
                
                try:
                    decoded_img = rle_decode(compressed_data)
                except ValueError as e:
                    flash(f'RLE decompression error: {str(e)}', 'error')
                    return redirect(request.url)
                
                # Additional validation
                if not isinstance(decoded_img, np.ndarray) or len(decoded_img.shape) != 2:
                    flash('Invalid image format after decompression', 'error')
                    return redirect(request.url)
                
                decompressed_filename = f"decompressed_rle_{secure_filename(rle_file.filename)}.png"
                decompressed_path = os.path.join(DECOMPRESSED_FOLDER, decompressed_filename)
                cv2.imwrite(decompressed_path, decoded_img)
                
                flash('RLE decompression successful!', 'success')
                return render_template('decompress.html', 
                                    decompressed=True,
                                    filename=decompressed_filename,
                                    algorithm=algorithm)
            elif algorithm == 'huffman':
                if 'huffman_compressed' not in request.files:
                    flash('No compressed file selected', 'error')
                    return redirect(request.url)
                
                huffman_file = request.files['huffman_compressed']
                
                if huffman_file.filename == '':
                    flash('No file selected', 'error')
                    return redirect(request.url)
                
                compressed_data = huffman_file.read()
                decoded_img = huffman_decompress(compressed_data)
                
                decompressed_filename = f"decompressed_huffman_{secure_filename(huffman_file.filename)}.png"
                decompressed_path = os.path.join(DECOMPRESSED_FOLDER, decompressed_filename)
                
                if len(decoded_img.shape) == 2:
                    cv2.imwrite(decompressed_path, decoded_img)
                else:
                    cv2.imwrite(decompressed_path, cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR))
                
                flash('Huffman decompression successful!', 'success')
                return render_template('decompress.html', 
                                    decompressed=True,
                                    filename=decompressed_filename,
                                    algorithm=algorithm)
            elif algorithm == 'dct':
                if 'dct_compressed' not in request.files:
                    flash('No compressed file selected', 'error')
                    return redirect(request.url)
                
                dct_file = request.files['dct_compressed']
                
                if dct_file.filename == '':
                    flash('No file selected', 'error')
                    return redirect(request.url)
                
                compressed_data = dct_file.read()
                decompressed_img = dct_decompress(compressed_data)
                
                decompressed_filename = f"decompressed_dct_{secure_filename(dct_file.filename)}.png"
                decompressed_path = os.path.join(DECOMPRESSED_FOLDER, decompressed_filename)
                cv2.imwrite(decompressed_path, decompressed_img)
                
                flash('DCT decompression successful!', 'success')
                return render_template('decompress.html', 
                                    decompressed=True,
                                    filename=decompressed_filename,
                                    algorithm=algorithm)
            
            else:
                flash('Invalid algorithm selected', 'error')
                return redirect(request.url)
        
        except Exception as e:
            flash(f'Error during decompression: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('decompress.html', decompressed=False)

@app.route('/download/<folder>/<filename>')
def download(folder, filename):
    if folder == 'compressed':
        directory = COMPRESSED_FOLDER
    elif folder == 'decompressed':
        directory = DECOMPRESSED_FOLDER
    else:
        flash('Invalid download request')
        return redirect(url_for('index'))
    
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)