import cv2

def compare_similarity(img1, img2):
    # Need to convert the to uint8
    img1 = img1.astype('uint8')
    img1_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])

    # Need to convert the to uint8
    img2 = img2.astype('uint8')
    img2_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # compare the histogram using Bhattacharyya distance
    # https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist
    # the smaller the better
    dist = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
    return dist, img1_hist, img2_hist