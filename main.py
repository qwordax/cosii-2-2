import cv2 as cv
import numpy as np

def gamma_correction(image, a, gamma):
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            new_pixel = int(a * pow(image[i, j] / 255, gamma) * 255)

            if new_pixel > 255:
                new_pixel = 255

            image[i, j] = new_pixel

    return image

def erosion(image, kernel):
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape

    k_rows_2, k_cols_2 = k_rows // 2, k_cols // 2

    result = image.copy()

    for i in range(k_rows_2, rows - k_rows_2):
        for j in range(k_cols_2, cols - k_cols_2):
            min_value = 255

            for k in range(k_rows):
                for l in range(k_cols):
                    pixel = image[i-k_rows_2+k, j-k_cols_2+l]
                    k_value = kernel[k, l]
                    min_value = min(min_value, pixel*k_value)

            result[i, j] = min_value

    return result

def dilation(image, kernel):
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape

    k_rows_2, k_cols_2 = k_rows // 2, k_cols // 2

    result = image.copy()

    for i in range(k_rows_2, rows - k_rows_2):
        for j in range(k_cols_2, cols - k_cols_2):
            max_value = 0

            for k in range(k_rows):
                for l in range(k_cols):
                    pixel = image[i-k_rows_2+k, j-k_cols_2+l]
                    k_value = kernel[k, l]
                    max_value = max(max_value, pixel*k_value)

            result[i, j] = max_value

    return result

def threshold(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size

    best_threshold = 0
    best_variance = 0

    for t in range(1, 256):
        if np.sum(hist[:t]) == 0 or np.sum(hist[t:]) == 0:
            continue

        w0 = np.sum(hist[:t])/total_pixels
        w1 = np.sum(hist[t:])/total_pixels

        u0 = np.sum(np.arange(t)*hist[:t]) / (w0*total_pixels)
        u1 = np.sum(np.arange(t, 256)*hist[t:]) / (w1*total_pixels)

        variance = w0 * w1 * (u0-u1) ** 2

        if variance > best_variance:
            best_threshold = t
            best_variance = variance

    return (image > best_threshold).astype(np.uint8) * 255

def components(image):
    pass

def k_means(image, k):
    pass

def main():
    kernel = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]], dtype=np.uint8)

    path = input('path: ')
    k = int(input('k: '))

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    cv.imshow('Initial Image', image)

    image = gamma_correction(image, 2, 6)
    image = erosion(image, kernel)
    # image = dilation(image, kernel)
    image = threshold(image)

    cv.imshow('Binary Image', image)

    components(image)
    k_means(image, k)

    cv.waitKey(0)

if __name__ == "__main__":
    main()
