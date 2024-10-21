import numpy as np
import cv2 as cv

from typing import Tuple


def mask_hsv(img, color_params=None):
    imageHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    minHsv = np.array([color_params[0][0], color_params[1][0], color_params[2][0]])
    maxHsv = np.array([color_params[0][1], color_params[1][1], color_params[2][1]])

    h, w = imageHsv.shape[:2]
    mask = cv.inRange(imageHsv, minHsv, maxHsv)
    # img_masked = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8) * 255
    return None, mask


# def masking_hsv():

#     SET_MASK_WINDOW = "Set Mask"
#     cv.namedWindow(SET_MASK_WINDOW)

#     minHue = 0
#     maxHue = 34
#     minSat = 120
#     maxSat = 255
#     minVal = 161
#     maxVal = 255
#     cv.createTrackbar("Min Hue", SET_MASK_WINDOW, minHue, 179, noop)
#     cv.createTrackbar("Max Hue", SET_MASK_WINDOW, maxHue, 179, noop)
#     cv.createTrackbar("Min Sat", SET_MASK_WINDOW, minSat, 255, noop)
#     cv.createTrackbar("Max Sat", SET_MASK_WINDOW, maxSat, 255, noop)
#     cv.createTrackbar("Min Val", SET_MASK_WINDOW, minVal, 255, noop)
#     cv.createTrackbar("Max Val", SET_MASK_WINDOW, maxVal, 255, noop)

#      ## 2. Read and convert image to HSV color space
#     image = cv.imread("imgs/sub_edge_case0.png")
#     #image = undistort_img(raw)
#     imageHsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     width  = image.shape[1]
#     height = image.shape[0]
#     size_ratio = 0.5

#     winIn = cv.namedWindow("in", cv.WINDOW_NORMAL)
#     cv.resizeWindow("in", int(width * size_ratio), int(height * size_ratio))

#     winOut = cv.namedWindow("out", cv.WINDOW_NORMAL)
#     cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))

#     while True:


#         ## 3. Get min and max HSV values from Set Mask window
#         minHue = cv.getTrackbarPos("Min Hue", SET_MASK_WINDOW)
#         maxHue = cv.getTrackbarPos("Max Hue", SET_MASK_WINDOW)
#         minSat = cv.getTrackbarPos("Min Sat", SET_MASK_WINDOW)
#         maxSat = cv.getTrackbarPos("Max Sat", SET_MASK_WINDOW)
#         minVal = cv.getTrackbarPos("Min Val", SET_MASK_WINDOW)
#         maxVal = cv.getTrackbarPos("Max Val", SET_MASK_WINDOW)
#         minHsv = np.array([minHue, minSat, minVal])
#         maxHsv = np.array([maxHue, maxSat, maxVal])

#         ## 4. Create mask and result (masked) image
#         # params: input array, lower boundary array, upper boundary array
#         mask = cv.inRange(imageHsv, minHsv, maxHsv)
#         cv.imwrite("imgs/mask.jpg", mask)

#         # params: src1 array, src2 array, mask
#         resultImage = cv.bitwise_and(image, image, mask=mask)
#         cropped = resultImage[:, :]
#         cv.imwrite("imgs/feature.jpg", cropped)

#         ## 5. Show images
#         win = cv.namedWindow("out", cv.WINDOW_NORMAL)
#         # size_ratio = 5
#         # height, width,_ = resultImage.shape
#         # cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))
#         #cv.resizeWindow("out", 1000,1000)
#         cv.imshow("in", image)
#         # cv.imshow("Mask", mask)   # optional
#         cv.imshow("out", resultImage)
#         resultImage = resultImage.astype(np.uint8)
#         cv.imwrite("imgs/masked.jpg", resultImage)
#         if cv.waitKey(1) == 27: break   # Wait Esc key to end program


# def save_hsv():
#     image = cv.imread("img_proc_python/imgs/sub.jpg")
#     img_masked  = mask_hsv(image)
#     cv.imwrite("imgs/disk_masked.jpg", img_masked)
#     #cv.imwrite("imgs/disk.jpg", img_processed)

# def mask_hsb(img):
#     brightness = np.sum(img, axis = 2) / 3
#     mask = (brightness > 90).astype(float) * 255
#     img_masked = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
#     return img_masked, img_masked

# def save_hsb():
#     image = cv.imread("imgs/sub_hsv.jpg")
#     img_masked, img_processed = mask_hsb(image)
#     cv.imwrite("imgs/sub_masked_hsb.jpg", img_masked)
#     cv.imwrite("imgs/sub_hsb.jpg", img_processed)

# def noop(x):
#     pass

# def masking_hsv_by():
#     SET_MASK_WINDOW = "Set Mask"
#     cv.namedWindow(SET_MASK_WINDOW)

#     minHue = 0
#     maxHue = 34
#     minSat = 120
#     maxSat = 255
#     minVal = 161
#     maxVal = 255
#     cv.createTrackbar("Min Hue", SET_MASK_WINDOW, minHue, 179, noop)
#     cv.createTrackbar("Max Hue", SET_MASK_WINDOW, maxHue, 179, noop)
#     cv.createTrackbar("Min Sat", SET_MASK_WINDOW, minSat, 255, noop)
#     cv.createTrackbar("Max Sat", SET_MASK_WINDOW, maxSat, 255, noop)
#     cv.createTrackbar("Min Val", SET_MASK_WINDOW, minVal, 255, noop)
#     cv.createTrackbar("Max Val", SET_MASK_WINDOW, maxVal, 255, noop)

#      ## 2. Read and convert image to HSV color space
#     image = cv.imread("imgs/edge_case0.png")
#     #image = undistort_img(raw)
#     imageHsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     width  = image.shape[1]
#     height = image.shape[0]
#     size_ratio = 0.5

#     winIn = cv.namedWindow("in", cv.WINDOW_NORMAL)
#     cv.resizeWindow("in", int(width * size_ratio), int(height * size_ratio))

#     winOut = cv.namedWindow("out", cv.WINDOW_NORMAL)
#     cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))

#     while True:


#         ## 3. Get min and max HSV values from Set Mask window
#         minHue = cv.getTrackbarPos("Min Hue", SET_MASK_WINDOW)
#         maxHue = cv.getTrackbarPos("Max Hue", SET_MASK_WINDOW)
#         minSat = cv.getTrackbarPos("Min Sat", SET_MASK_WINDOW)
#         maxSat = cv.getTrackbarPos("Max Sat", SET_MASK_WINDOW)
#         minVal = cv.getTrackbarPos("Min Val", SET_MASK_WINDOW)
#         maxVal = cv.getTrackbarPos("Max Val", SET_MASK_WINDOW)
#         minHsv = np.array([minHue, minSat, minVal])
#         maxHsv = np.array([maxHue, maxSat, maxVal])

#         ## 4. Create mask and result (masked) image
#         # params: input array, lower boundary array, upper boundary array
#         mask = cv.inRange(imageHsv, minHsv, maxHsv)
#         cv.imwrite("imgs/mask.jpg", mask)

#         # params: src1 array, src2 array, mask
#         resultImage = cv.bitwise_and(image, image, mask=mask)
#         cropped = resultImage[:, :]
#         cv.imwrite("imgs/feature.jpg", cropped)

#         ## 5. Show images
#         win = cv.namedWindow("out", cv.WINDOW_NORMAL)
#         # size_ratio = 5
#         # height, width,_ = resultImage.shape
#         # cv.resizeWindow("out", int(width * size_ratio), int(height * size_ratio))
#         #cv.resizeWindow("out", 1000,1000)
#         cv.imshow("in", image)
#         # cv.imshow("Mask", mask)   # optional
#         cv.imshow("out", resultImage)
#         resultImage = resultImage.astype(np.uint8)
#         cv.imwrite("imgs/masked.jpg", resultImage)
#         if cv.waitKey(1) == 27: break   # Wait Esc key to end program
