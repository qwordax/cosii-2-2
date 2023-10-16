import cv2 as cv

def gamma_correction():
    pass

def erosion():
    pass

def dilation():
    pass

def main():
    path = input('path: ')
    k = int(input('k: '))

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)

    cv.imshow('Initial Image', image)

    cv.waitKey(0)

if __name__ == "__main__":
    main()
