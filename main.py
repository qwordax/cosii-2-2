import cv2 as cv

def gamma_correction(image):
    return image

def erosion(image):
    return image

def dilation(image):
    return image

def threshold(image):
    return image

def components(image):
    pass

def main():
    path = input('path: ')
    k = int(input('k: '))

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    cv.imshow('Initial Image', image)

    image = gamma_correction(image)
    image = erosion(image)
    image = dilation(image)
    image = threshold(image)

    cv.imshow('Binary Image', image)

    components(image)

    cv.waitKey(0)

if __name__ == "__main__":
    main()
