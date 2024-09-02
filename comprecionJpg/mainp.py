import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb_to_ycbcr(image):
    """
    Convierte una imagen de espacio de color RGB a YCbCr.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

def downsample(image_ycbcr):
    """
    Realiza el submuestreo 4:2:0 en una imagen YCbCr.
    """
    Y, Cb, Cr = cv2.split(image_ycbcr)
    
    # Reducir la resolución de Cb y Cr
    Cb = cv2.pyrDown(Cb)
    Cr = cv2.pyrDown(Cr)
    
    return Y, Cb, Cr

def split_into_blocks(channel):
    """
    Divide una imagen (o canal) en bloques de 8x8 píxeles.
    """
    height, width = channel.shape
    blocks = []

    # Iterar sobre la imagen en pasos de 8 píxeles para extraer bloques de 8x8.
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = channel[i:i+8, j:j+8]
            if block.shape == (8, 8):
                blocks.append(block)

    return blocks

def dct_2d(block):
    """
    Aplica la Transformada Discreta del Coseno (DCT) a un bloque de 8x8 píxeles.
    """
    return cv2.dct(np.float32(block) - 128)

def quantize(block, quant_matrix):
    """
    Cuantiza un bloque de coeficientes DCT usando una matriz de cuantización.
    """
    return np.round(block / quant_matrix)

def inverse_quantize(block, quant_matrix):
    """
    Realiza la cuantización inversa multiplicando el bloque de coeficientes 
    cuantizados por la matriz de cuantización.
    """
    return block * quant_matrix

def idct_2d(block):
    """
    Aplica la Transformada Inversa del Coseno Discreto (IDCT) a un bloque 
    de coeficientes DCT cuantizados para obtener el bloque espacial.
    """
    return cv2.idct(block) + 128

def combine_blocks(blocks, height, width):
    """
    Combina bloques de 8x8 píxeles en una imagen de tamaño especificado.
    """
    num_blocks_y = height // 8
    num_blocks_x = width // 8
    
    image = np.zeros((height, width), dtype=np.float32)
    
    block_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if block_idx < len(blocks):
                image[i:i+8, j:j+8] = blocks[block_idx]
                block_idx += 1
            else:
                raise IndexError("Número de bloques es menor que esperado")
    
    return image

def save_as_jpeg(y, cb, cr, output_filename):
    """
    Guarda una imagen YCbCr como un archivo JPEG.
    """
    # Redimensionar Cb y Cr a las dimensiones de Y
    cb_resized = cv2.resize(cb, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    cr_resized = cv2.resize(cr, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)

    ycbcr = cv2.merge([y, cb_resized, cr_resized])
    rgb = cv2.cvtColor(np.uint8(ycbcr), cv2.COLOR_YCrCb2RGB)
    image = Image.fromarray(rgb)
    image.save(output_filename, "JPEG")

def pad_image(image):
    """
    Añade padding a la imagen para que sus dimensiones sean múltiplos de 8.
    """
    h, w = image.shape[:2]
    h_pad = (8 - (h % 8)) % 8
    w_pad = (8 - (w % 8)) % 8
    
    padded_image = cv2.copyMakeBorder(image, 0, h_pad, 0, w_pad, cv2.BORDER_REPLICATE)
    
    return padded_image

def plot_images(original_image, reconstructed_image, dct_blocks, final_image):
    """
    Muestra la imagen original, la imagen reconstruida, bloques DCT y la imagen final reconstruida.
    
    Parámetros:
    - original_image: Imagen original en formato RGB.
    - reconstructed_image: Imagen reconstruida a partir de los bloques DCT.
    - dct_blocks: Lista de bloques DCT a mostrar.
    - final_image: Imagen final reconstruida en formato YCbCr convertido a RGB.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Mostrar la imagen original
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Imagen Original")
    axes[0, 0].axis('off')
    
    # Mostrar la imagen reconstruida
    axes[0, 1].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_YCrCb2RGB))
    axes[0, 1].set_title("Imagen Reconstruida")
    axes[0, 1].axis('off')
    
    # Mostrar la imagen final reconstruida
    axes[0, 2].imshow(final_image)
    axes[0, 2].set_title("Imagen Final Reconstruida")
    axes[0, 2].axis('off')
    
    # Mostrar los primeros 4 bloques DCT
    for i, block in enumerate(dct_blocks):
        ax = axes[1, i]
        ax.imshow(np.log(np.abs(block) + 1), cmap='gray')  # Usar escala logarítmica para mejor visualización
        ax.set_title(f'Bloque DCT {i+1}')
        ax.axis('off')

    # Mostrar la imagen final reconstruida
    axes[2, 0].imshow(final_image)
    axes[2, 0].set_title("Imagen Final")
    axes[2, 0].axis('off')
    
    # Mostrar imagen vacía en la última posición
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


# Cargar la imagen de archivo
image = cv2.imread('a.jpg')

# Aplicar padding a la imagen para asegurar dimensiones múltiplos de 8
image = pad_image(image)

# Convertir la imagen de BGR a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir la imagen RGB a YCbCr
image_ycbcr = rgb_to_ycbcr(image_rgb)

# Realizar el submuestreo de crominancia
Y, Cb, Cr = downsample(image_ycbcr)

# Dividir cada canal en bloques de 8x8 píxeles
Y_blocks = split_into_blocks(Y)
Cb_blocks = split_into_blocks(Cb)
Cr_blocks = split_into_blocks(Cr)

# Aplicar la Transformada Discreta del Coseno (DCT) a cada bloque
Y_dct = [dct_2d(block) for block in Y_blocks]
Cb_dct = [dct_2d(block) for block in Cb_blocks]
Cr_dct = [dct_2d(block) for block in Cr_blocks]

# Matrices de cuantización estándar
quantization_matrix_luminance = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

quantization_matrix_chrominance = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

# Cuantizar los bloques DCT
Y_quantized = [quantize(block, quantization_matrix_luminance) for block in Y_dct]
Cb_quantized = [quantize(block, quantization_matrix_chrominance) for block in Cb_dct]
Cr_quantized = [quantize(block, quantization_matrix_chrominance) for block in Cr_dct]

# Realizar la cuantización inversa
Y_dequantized = [inverse_quantize(block, quantization_matrix_luminance) for block in Y_quantized]
Cb_dequantized = [inverse_quantize(block, quantization_matrix_chrominance) for block in Cb_quantized]
Cr_dequantized = [inverse_quantize(block, quantization_matrix_chrominance) for block in Cr_quantized]

# Aplicar la Transformada Inversa del Coseno Discreto (IDCT)
Y_idct = [idct_2d(block) for block in Y_dequantized]
Cb_idct = [idct_2d(block) for block in Cb_dequantized]
Cr_idct = [idct_2d(block) for block in Cr_dequantized]

# Combinar los bloques IDCT en una imagen
Y_combined = combine_blocks(Y_idct, Y.shape[0], Y.shape[1])
Cb_combined = cv2.resize(combine_blocks(Cb_idct, Cb.shape[0], Cb.shape[1]), (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
Cr_combined = cv2.resize(combine_blocks(Cr_idct, Cr.shape[0], Cr.shape[1]), (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)

# Guardar la imagen reconstruida
save_as_jpeg(Y_combined, Cb_combined, Cr_combined, 'output.jpg')





plot_images(image_rgb, cv2.merge([Y_combined, Cb_combined, Cr_combined]), Y_dct[:4], cv2.merge([Y_combined, Cb_combined, Cr_combined]))
