**JPEG Compression using DCT**
================================

This is a Python implementation of JPEG compression using the Discrete Cosine Transform (DCT). The code compresses an image using the DCT, quantizes the coefficients, and then reconstructs the image using inverse DCT and quantization.

**Functions**
-------------

### `rgb_to_ycbcr(image)`

Converts an RGB image to YCbCr color space.

### `downsample(image_ycbcr)`

Performs 4:2:0 chroma subsampling on a YCbCr image.

### `split_into_blocks(channel)`

Divides a channel into 8x8 blocks.

### `dct_2d(block)`

Applies the 2D Discrete Cosine Transform (DCT) to a block.

### `quantize(block, quant_matrix)`

Quantizes a block using a quantization matrix.

### `inverse_quantize(block, quant_matrix)`

Performs inverse quantization of a block using a quantization matrix.

### `idct_2d(block)`

Applies the 2D Inverse Discrete Cosine Transform (IDCT) to a block.

### `combine_blocks(blocks, height, width)`

Combines blocks into a single image.

### `save_as_jpeg(y, cb, cr, output_filename)`

Saves a YCbCr image as a JPEG file.

### `pad_image(image)`

Adds padding to an image to make its dimensions multiples of 8.

### `plot_images(original_image, reconstructed_image, dct_blocks, final_image)`

Plots the original image, reconstructed image, DCT blocks, and final reconstructed image.

**Usage**
---------

1. Load an image using `cv2.imread`.
2. Apply padding to the image using `pad_image`.
3. Convert the image from BGR to RGB using `cv2.cvtColor`.
4. Convert the image from RGB to YCbCr using `rgb_to_ycbcr`.
5. Perform chroma subsampling using `downsample`.
6. Divide each channel into 8x8 blocks using `split_into_blocks`.
7. Apply the DCT to each block using `dct_2d`.
8. Quantize the DCT coefficients using `quantize`.
9. Perform inverse quantization using `inverse_quantize`.
10. Apply the IDCT to each block using `idct_2d`.
11. Combine the blocks into a single image using `combine_blocks`.
12. Save the reconstructed image as a JPEG file using `save_as_jpeg`.
13. Plot the original image, reconstructed image, DCT blocks, and final reconstructed image using `plot_images`.

**License**
---------

This implementation is for educational purposes only and is not intended for commercial use.
