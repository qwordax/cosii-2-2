import cv2 as cv
import numpy as np
import warnings

def gamma_correction(image, a, gamma):
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            new_pixel = int(a * pow(image[i, j]/255, gamma) * 255)

            if new_pixel > 255:
                new_pixel = 255

            image[i, j] = new_pixel

    return image

def erosion(image, kernel):
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape

    k_rows_2, k_cols_2 = k_rows//2, k_cols//2

    result = image.copy()

    for i in range(k_rows_2, rows-k_rows_2):
        for j in range(k_cols_2, cols-k_cols_2):
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

    k_rows_2, k_cols_2 = k_rows//2, k_cols//2

    result = image.copy()

    for i in range(k_rows_2, rows-k_rows_2):
        for j in range(k_cols_2, cols-k_cols_2):
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

def params(component):
    rows, cols = component.shape

    square = 0
    perimeter = 0
    compactness = 0
    elongation = 0
    static_x = 0
    static_y = 0

    m11 = 0
    m02 = 0
    m20 = 0

    for i in range(rows):
        for j in range(cols):
            if component[i, j] == 255:
                square += 1
                static_x += i
                static_y += j

                if (i == 0 or i == rows-1 or
                    j == 0 or j == cols-1):
                    perimeter += 1
                elif (component[i, j+1] == 0 or
                      component[i, j-1] == 0 or
                      component[i+1, j] == 0 or
                      component[i-1, j] == 0):
                    perimeter += 1

    for i in range(rows):
        for j in range(cols):
            if component[i, j] == 255:
                m11 += (i - static_x/square) * (j - static_y/square)
                m02 += (i - static_x/square) ** 2
                m20 += (j - static_y/square) ** 2

    compactness = perimeter * perimeter / square

    elongation = ((m20 + m02 + np.sqrt((m20-m02) ** 2 + 4*m11 ** 2)) /
                  (m20 + m02 - np.sqrt((m20-m02) ** 2 + 4*m11 ** 2)))

    return [square, perimeter, compactness, elongation, static_x, static_y]

def components(image):
    data = []

    rows, cols = image.shape
    index = 1

    labeled = np.zeros_like(image, dtype=int)

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255 and labeled[i, j] == 0:
                component = np.zeros((rows, cols), dtype=np.uint8)

                stack = [(i, j)]

                while stack:
                    x, y = stack.pop()

                    if (x >= 0 and x < rows and
                        y >= 0 and y < cols and
                        image[x, y] == 255 and
                        labeled[x, y] == 0):
                        labeled[x, y] = index
                        component[x, y] = 255

                        stack.extend([(x+1, y),
                                      (x-1, y),
                                      (x, y+1),
                                      (x, y-1)])

                data_row = params(component);

                if data_row[0] > 200:
                    data.append(data_row[:4])

                    print()
                    print('%11s: %d' % ('index', index))
                    print('%11s: %d' % ('square', data_row[0]))
                    print('%11s: %d' % ('perimeter', data_row[1]))
                    print('%11s: %.2f' % ('compactness', data_row[2]))
                    print('%11s: %.2f' % ('elongation', data_row[3]))
                    print('%11s: %d' % ('static_x', data_row[4]))
                    print('%11s: %d' % ('static_y', data_row[5]))

                    cv.imshow(str(index), component); index += 1

    return np.array(data)

def distance(a, b):
    return np.sqrt(np.sum((a-b) ** 2))

def k_means(data, k):
    rows, _ = data.shape

    cluster_centers = data[np.random.choice(rows, k, replace=False)]
    cluster_assigns = np.zeros(rows)

    for _ in range(10):
        for i, row in enumerate(data):
            distances = [distance(row, center) for center in cluster_centers]
            closest_cluster = np.argmin(distances)
            cluster_assigns[i] = closest_cluster

        for cluster in range(k):
            cluster_rows = data[cluster_assigns == cluster]

            if len(cluster_rows) > 0:
                cluster_centers[cluster] = np.mean(cluster_rows, axis=0)

    print()

    for i in range(k):
        cluster_components = ([j+1 for j, cluster in enumerate(cluster_assigns)
                               if cluster == i])

        print(f'{i+1}: {cluster_components}')

def main():
    warnings.filterwarnings('ignore')

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

    k_means(components(image), k)

    cv.waitKey(0)

if __name__ == "__main__":
    main()
