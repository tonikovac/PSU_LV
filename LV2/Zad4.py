import matplotlib.pyplot as plt
import numpy as np

def blackWhiteField(pixel_number, square_number_width, square_number_height):
    black = np.zeros((pixel_number, pixel_number), dtype=np.uint8)
    white = np.ones((pixel_number, pixel_number), dtype=np.uint8) * 255
    matrix = []

    for i in range(square_number_width):
        row = []
        for j in range(square_number_height):
            if (i + j) % 2 == 0:
                row.append(black)
            else:
                row.append(white)
        matrix.append(np.hstack(row))

    img = np.vstack(matrix)
    return img

img = blackWhiteField(50, 10, 10)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()