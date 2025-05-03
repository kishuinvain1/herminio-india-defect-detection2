import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64





def load_image():
    opencv_image_resz = None
    opencv_image = None 
    path = None
    f = None
    uploaded_file = st.file_uploader(label='Pick an image to test')
    print(uploaded_file)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image_resz = cv2.resize(opencv_image, (1024, 1024))
        image_data = uploaded_file.getvalue() 
        #st.image(image_data)
        name = uploaded_file.name
        path = os.path.abspath(name)
        print("abs path")
        print(path)
	
        cv2.imwrite("main_image.jpg", opencv_image_resz)
       
    return path, opencv_image_resz
       


	


# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return np.array(image)
	

	
def drawBoundingBox(saved_image ,x, y, w, h, cl, cf):
    #img = Image.open(saved_image)
    #img = cv2.imread(saved_image)
    #img = cv2.cvtColor(saved_image,cv2.COLOR_BGR2RGB)
    img = saved_image
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    start_pnt = (x-w//2-15,y-h//2-15)
    end_pnt = (x+w//2+15, y+h//2+15)
    txt_start_pnt = (x-w//2, y-h//2-15-15)

    if cl == "Ok":
        color = (0,255,0)
    else:
        color = (255,0,0)    
        
    img = cv2.rectangle(img, start_pnt, end_pnt, color, 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)	
    return img
    	
    


def predict(model, url):
    return model.predict(url, confidence=50, overlap=70).json()
    #return model.predict(url, hosted=True).json()
	
	
def main():
    st.title('Defect Detection v2')

    rf = Roboflow(api_key="0N1hjNKtBabtHfuP93Q8")
    project = rf.workspace("verify-gn-hnnai").project("herminio-object-detection")
    model = project.version(1).model
    
                
    image, svd_img = load_image()

    result = st.button('Detect')
    if result:
        results = predict(model, svd_img)
        #results = predict(model2, url)
        print("Prediction Results are...")	
        print(results)
        if len(results['predictions']) == 0:
            st.image(svd_img)
            st.write("No Object Detected")
        else:
            roi_count = 0
            for i in range(len(results['predictions'])):
                roi_count += 1
                new_img_pth = results['predictions'][i]['image_path']
                x = results['predictions'][i]['x']
                y = results['predictions'][i]['y']
                w = results['predictions'][i]['width']
                h = results['predictions'][i]['height']
                cl = results['predictions'][i]['class']
                cnf = results['predictions'][i]['confidence']
                x1 = int(x - w//2)
                x2 = int(x + w//2)
                y1 = int(y - h//2)
                y2 = int(y + h//2)
                roi = svd_img[y1:y2, x1:x2, :]
                cv2.imwrite(f"roi_{str(roi_count)}.jpg", roi)




                #svd_img = drawBoundingBox(svd_img,x, y, w, h, cl, cnf)

            st.write('DETECTION RESULTS')    
            st.image(svd_img, caption='Resulting Image')
            for i in range(roi_count):
                st.image(f"roi_{str(i)}.jpg", caption=f'roi_{str(i)}')
           

if __name__ == '__main__':
    main()
