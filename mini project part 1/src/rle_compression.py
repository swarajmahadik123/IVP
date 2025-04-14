import numpy as np  

def rle_encode(img):  
    flat_img = img.flatten()  
    encoded = []  
    count = 1  
    for i in range(1, len(flat_img)):  
        if flat_img[i] == flat_img[i-1]:  
            count += 1  
        else:  
            encoded.append((flat_img[i-1], count))  
            count = 1  
    encoded.append((flat_img[-1], count))  
    return encoded  

def rle_decode(encoded, shape):  
    img = []  
    for val, count in encoded:  
        img.extend([val] * count)  
    return np.array(img).reshape(shape)  