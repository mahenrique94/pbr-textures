import cv2
import numpy as np
from PIL import Image


def gen_normal_map(base_texture):
    # Load the base texture and convert to grayscale
    gray = cv2.cvtColor(np.array(base_texture), cv2.COLOR_BGR2GRAY)

    # Compute the Sobel gradients in X and Y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Normalize gradients to range [0, 1]
    sobelx = cv2.normalize(sobelx, None, 0, 1, cv2.NORM_MINMAX)
    sobely = cv2.normalize(sobely, None, 0, 1, cv2.NORM_MINMAX)

    # Prepare the normal map channels
    height, width = gray.shape
    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    # Set the Red channel to X gradient, Green to Y gradient, and Blue to the Z component
    normal_map[:, :, 0] = sobelx * 0.5 + 0.5  # R channel (X gradient)
    normal_map[:, :, 1] = sobely * 0.5 + 0.5  # G channel (Y gradient)
    normal_map[:, :, 2] = 1.0  # B channel (Z component, usually set to 1 for a flat surface)

    # Convert to 8-bit format for saving/displaying
    normal_map = (normal_map * 255).astype(np.uint8)

    # Save the normal map
    normal_map_image = Image.fromarray(normal_map)
    normal_map_image.save('normal_map.png')


def gen_roughness_map(base_texture):
    gray = cv2.cvtColor(np.array(base_texture), cv2.COLOR_BGR2GRAY)
    roughness_map = gray.copy()
    roughness_map = cv2.equalizeHist(roughness_map)
    roughness_map = Image.fromarray(roughness_map)
    roughness_map.save('roughness_map.png')


def gen_metallic_map(base_texture):
    gray = cv2.cvtColor(np.array(base_texture), cv2.COLOR_BGR2GRAY)
    metallic_map = np.zeros_like(gray)
    metallic_map[gray > 128] = 255  # Just an example threshold
    metallic_map = Image.fromarray(metallic_map)
    metallic_map.save('metallic_map.png')


def gen_ambient_occlusion_map(base_texture):
    gray = cv2.cvtColor(np.array(base_texture), cv2.COLOR_BGR2GRAY)
    ao_map = cv2.bitwise_not(gray)
    ao_map = cv2.GaussianBlur(ao_map, (21, 21), 0)
    ao_map = Image.fromarray(ao_map)
    ao_map.save('ambient_occlusion_map.png')


def main():
    base_texture = Image.open("base.png")
    gen_normal_map(base_texture)
    gen_roughness_map(base_texture)
    gen_metallic_map(base_texture)
    gen_ambient_occlusion_map(base_texture)


if __name__ == "__main__":
    main()
