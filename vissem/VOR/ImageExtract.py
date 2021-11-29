
import cv2

# 將輸入圖像分割為 ROI感兴趣区域
# 將圖像轉換為灰度、應用閾值、輪廓和邊界框。
def extract_elements(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # 閾值threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=13)
    # 膨胀函数dilate
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 使用轮廓判别法get contours
    # 对于发现的每个轮廓，在图上画一个矩形围绕它

    bboxes = []
    idx = 0

    # 然后把矩形的数值计算出来，即得到他们的roi
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append([x, y, w, h])

    # print(bboxes)类似：[[43, 195, 116, 38], [44, 80, 108, 109], [102, 67, 58, 58], [25, 27, 144, 57]]
    return bboxes
