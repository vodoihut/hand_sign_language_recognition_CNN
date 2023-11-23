import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) # Mở kết nối với camera
detector = HandDetector(maxHands=1) #Sử dụng thư viện để phát hiện bàn tay

offset = 20 # Khoảng cách lề từ phần tay phát hiện được
imgSize = 300 # Kích thước ảnh

folder = "datatrain/zZ" # Thư mục lưu ảnh
counter = 0 # Biến đếm số lượng ảnh đã lưu

while True:
    success, img = cap.read() # Đọc hình ảnh từ camer
    hands, img = detector.findHands(img) # Phát hiện bàn tay trong hình ảnh
    if hands: # Nếu có bàn tay được phát hiện
        hand = hands[0] # Lấy thông tin về bàn tay đầu tiên
        x, y, w, h = hand['bbox'] # Lấy thông tin về bounding box của bàn tay

        # Tạo ảnh để chứa phần tay cắt
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        # Cắt phần ảnh chứa bàn tay từ hình ảnh gốc
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    # Nhấn 's' để lưu ảnh
    # Nhấn 'q' để thoát khỏi chương trình
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/aA{counter}.jpg',imgWhite)
        print(counter)
    elif key == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
