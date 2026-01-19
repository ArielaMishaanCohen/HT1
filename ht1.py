import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_img(img, title="Imagen", cmap=None):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    # Matplotlib espera RGB, OpenCV carga BGR
    if len(img.shape) == 3 and cmap is None:
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_show = img
    plt.imshow(img_show, cmap=cmap)
    plt.axis('off')
    plt.show()

def manual_contrast_brightness(image, alpha, beta):
    """
    g(x) = alpha * f(x) + beta
    """
    # 1. Convertir a float y normalizar
    img = image.astype(np.float32) / 255.0

    # 2. Aplicar contraste y brillo
    img = alpha * img + (beta / 255.0)

    # 3. Limitar valores y volver a uint8
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    return img

def manual_gamma_correction(image, gamma):
    """
    Corrección gamma manual
    """
    # Normalizar
    img = image.astype(np.float32) / 255.0

    # Aplicar gamma
    img = np.power(img, gamma)

    # Volver a uint8
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    return img

def hsv_segmentation(image):
    """
    Segmentación HSV (color rosado como en el notebook)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Rangos usados en el ipynb (rosado)
    lower = np.array([160, 60, 60], dtype=np.uint8)
    upper = np.array([175, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    segmented = image.copy()
    segmented[mask == 0] = (0, 0, 0)

    return segmented

if __name__ == "__main__":
    img = cv2.imread("test.jpg")  # Cambia el nombre si es necesario

    if img is None:
        print("Error: No se encontró la imagen.")
    else:
        # 1. Contraste y brillo
        contrast_img = manual_contrast_brightness(img, 1.5, 20)
        show_img(contrast_img, "Contraste Alto (Manual)")

        # 2. Corrección gamma
        gamma_img = manual_gamma_correction(img, 0.5)
        show_img(gamma_img, "Corrección Gamma 0.5")

        # 3. Segmentación HSV
        seg_img = hsv_segmentation(img)
        show_img(seg_img, "Segmentación HSV (Rosado)")
