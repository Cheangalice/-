

import cv2
import numpy as np
import os
import Train
import ImageExtract

# classifier進行分類，得到labels
def classifier_classify(image):
    model_path = os.path.join("data/model/", "classifier.model")
    # 设置模型
    model = Train.ElementClassifier(model_path=model_path, load=True)
    labels = []
    # 调用imageExtract將輸入圖像分割為 ROI感兴趣区域
    # 在图像处理领域，感兴趣区域(ROI) 是从图像中选择的一个图像区域
    bboxes = ImageExtract.extract_elements(image)
    for bbox in bboxes:
        x, y, w, h = bbox
        roi = image[y - 20:y + h + 20, x - 20:x + w + 20]

        if np.size(roi) == 0:
            continue

        # 对roi进行预处理,得到预测结果数组
        pred = model.predict(roi, preprocess=True)
        indx = np.argmax(pred, axis=1)[0]

        # score取预测结果的数组里最大的那一个
        score = pred[0][indx]
        cv2.rectangle(image, (x-20, y-20), (x + w+20, y + h+20), (0, 191, 255), thickness=2)
        # label为预测结果集里score最高的那一项，即#{0: 'barchart', 1: 'button', 2: 'call', 3: 'circle', 4: 'triangle'}其中一个
        text = str(indx) + ":" + model.id2label[indx] + ":" + str(score)
        #cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
        #cv2.imshow("Input", image)
        #print(indx, model.id2label[indx], score)
        labels.append(model.id2label[indx])


    return labels
    cv2.waitKey(0)
    
