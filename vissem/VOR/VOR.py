import Train
import classify
import cosine_nlm
import cv2
import os
import classify
import numpy as np

# search_files里有没有用户输入的图形
def search_files(data_dir):
    #输入图形
    search_query = input("Please Enter your query: ")

    files = os.listdir(data_dir)
    count = 0
    
    for file in files: #循环要搜索的文件夹
      path = os.path.join(data_dir, file)
      image = cv2.imread(path)
      labels = classify.classifier_classify(image) #调用classify分类器来获得图像的labels
      text1 = []
      for label in labels:
        text1.append("This program produces " + label)  #标签被装入固定模板来生成句子，即图像描述
        
      cosines = []
      for text in text1:
        vector1 = cosine_nlm.text_to_vector(text)           #第一个向量为要搜索的文件夹里的labels的图像描述的集合
        vector2 = cosine_nlm.text_to_vector(search_query)   #第二个向量为用户输入的图形
        # {'This': 1, 'program': 1, 'produces': 1, 'triangle': 1}
        # {'call': 1}

        cosines.append(cosine_nlm.get_cosine(vector1, vector2))  #把两个向量作为cosines的参数
      
      # cosine >= 0.5时代表vector1和vectors有相同的图形表示
      for cosine in cosines:
         if cosine>=0.5 :       #当cosine >= 0.5 时，证明要搜索的文件夹里的label符合用户输入的图形
           count += 1
           print(count)
           print("programs found that produces output as" )
           print(search_query)
           #Also display path here
           break
                     
    if count == 0:
      print("No program found that matches your query!")



if __name__ == '__main__':
    #对database里的图形进行训练（训练后可以注释掉）、、
    #Train.classifier_train()

    #要在search_files中搜索我的输入
    search_files("C:/Users/Alice Cheang/Desktop/SoftwareTestingAndQuality/2021tool_implementation/vissem/data/test_images")
           

# Run: python classifier.py



