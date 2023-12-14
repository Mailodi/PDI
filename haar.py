import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image

def plot_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.ravel()
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def haar2D(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD

def reconstruct_haar2D(cA, cH, cV, cD):
    coeffs = cA, (cH, cV, cD)
    reconstructed_image = pywt.idwt2(coeffs, 'haar')
    return reconstructed_image

image_path = 'Lenna.png'
original_image = Image.open(image_path).convert('L')  # Converta para escala de cinza se não estiver

# Converta a imagem para uma matriz NumPy
image_array = np.array(original_image)

images_to_show = [image_array]
titles = ['Imagem Original']

# Aplicar a Transformada Discreta Bidimensional de Haar e armazenar cada estágio
cA, cH, cV, cD = haar2D(image_array)
images_to_show.extend([cA, cH, cV, cD])
titles.extend(['Aproximação (cA)', 'Horizontal (cH)', 'Vertical (cV)', 'Diagonal (cD)'])

# Reconstruir a imagem a partir dos coeficientes
reconstructed_image = reconstruct_haar2D(cA, cH, cV, cD)
images_to_show.append(reconstructed_image)
titles.append('Imagem Reconstruída')

plot_images(images_to_show, titles, 1, len(images_to_show))
