import cv2 as cv

def gamma_correction(image, a, gamma):
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            new_pixel = int(a * pow(image[i, j] / 255, gamma) * 255)

            if new_pixel > 255:
                new_pixel = 255

            image[i, j] = new_pixel

    return image

def erosion(image):
    return image

def dilation(image):
    return image

def threshold(image):
    return image

def components(image):
    pass

def k_means(image, k):
    pass

def main():
    path = input('path: ')
    k = int(input('k: '))

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    cv.imshow('Initial Image', image)

    image = gamma_correction(image, 2, 6)
    image = erosion(image)
    image = dilation(image)
    image = threshold(image)

    cv.imshow('Binary Image', image)

    components(image)
    k_means(image, k)

    cv.waitKey(0)

if __name__ == "__main__":
    main()
