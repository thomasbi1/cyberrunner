import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import time
from copy import deepcopy


def detect_gaussian(mask, j, q, th, show_sub, use_contour=True):
    if not use_contour:
        X = np.array(np.where(mask > 0)).T
        # print(X)
        if X.shape[0] < 2:
            return np.array([0, 0])  # $$ to change -> handle this case
        # X0 = deepcopy(X)
        if show_sub:
            X0 = deepcopy(X)

        k = 0
        while True:
            k += 1
            m = X.shape[0]
            mean = np.mean(X, axis=0)
            var = np.cov(X.T)
            if abs(np.linalg.det(var)) < 10 ** (-8):
                print("GAUSSIAN: SINGULAR MATRIX")
                return np.array([0, 0])  # $$ handle this case
            p = (
                1
                / (2 * np.pi * np.linalg.det(var) ** (0.5))
                * np.exp(
                    -0.5
                    * np.sum(((X - mean) @ np.linalg.inv(var)) * (X - mean), axis=1)
                )
            )
            perc = np.percentile(p, q)

            if np.min(p) >= th or k > 10:
                break

            X = X[p >= perc]

        c = mean

        if show_sub:
            im_out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype="uint8")

            for i in range(X0.shape[0]):
                im_out[X0[i, 0], X0[i, 1], 2] = 255

            for i in range(X.shape[0]):
                im_out[X[i, 0], X[i, 1], 1] = 255
                im_out[X[i, 0], X[i, 1], 2] = 0

            cv.drawMarker(
                im_out, c.astype(int)[::-1], (255, 0, 0), cv.MARKER_TILTED_CROSS, 5, 1
            )
            cv.imshow("sub_" + str(j), im_out)

    else:
        if j < 4:  # corners
            # cv.imshow("mask raw" + str(j), mask)

            # kernel_erosion = np.ones((3,3),np.uint8)
            # kernel_dilatation = np.ones((3,3),np.uint8)
            # erosion = cv.erode(mask,kernel_erosion,iterations = 1)
            # mask = cv.dilate(erosion,kernel_dilatation,iterations = 1)
            # cv.imshow("mask eroded"+ str(j), erosion)
            # cv.imshow("mask eroded and dilated"+ str(j), mask)
            # TODO maybe add some dilation
            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            if len(contours) == 0:
                c = (np.asarray(mask.shape) - 1.0) / 2.0
                blob_found = False
            else:
                contour = max(contours, key=cv.contourArea)
                M = cv.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    c = np.array([cy, cx])
                    blob_found = True
                else:
                    c = (np.asarray(mask.shape) - 1.0) / 2.0
                    blob_found = False

        else:  # ball
            # TODO maybe add some dilation
            # cv.imshow("mask raw", mask)

            kernel_erosion = np.ones((2, 2), np.uint8)
            kernel_dilatation = np.ones((5, 5), np.uint8)
            erosion = cv.erode(mask, kernel_erosion, iterations=1)
            mask = cv.dilate(erosion, kernel_dilatation, iterations=1)
            # mask_opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            # cv.imshow("mask eroded", erosion)
            # cv.imshow("mask eroded and dilated", mask)

            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            if len(contours) == 0:
                c = np.asarray(mask.shape) / 2.0
                blob_found = False
            else:
                contour = max(contours, key=cv.contourArea)
                area = cv.contourArea(contour)
                perimeter = cv.arcLength(contour, True)
                if perimeter == 0:
                    circularity = 0
                else:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)

                M = cv.moments(contour)

                # print("circularity", circularity)
                # print("area", area)
                # print("")
                if M["m00"] != 0 and circularity > 0.12 and area > 110:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    c = np.array([cy, cx])
                    blob_found = True
                    # print("circularity", circularity)
                    # print("center", c)
                else:
                    c = np.asarray(mask.shape) / 2.0
                    blob_found = False

        if show_sub:
            cv.imshow("sub_{}".format(j), mask)
            # cv.imshow("{}".format(j), mask)
            # for i in range(X.shape[0]):
            #     cv.drawMarker(img, (X[i,:])[::-1], (0,0,255), cv.MARKER_SQUARE, 5, 1)

    return c, blob_found


if __name__ == "__main__":

    im = cv.imread("imgs/out_true.jpg")

    c = detect_gaussian(im, 0, 5, 0.5, True)
    print(c)
    # cv.drawMarker(im, c, (0,0,255), cv.MARKER_TILTED_CROSS, 5, 1)

    cv.imshow("out", im)
    cv.waitKey(0)
