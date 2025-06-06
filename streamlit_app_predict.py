import io
import streamlit as st
from roboflow import Roboflow
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import base64


def resize_image(image, width=None, height=None):
    """Resizes an image while maintaining aspect ratio.

    Args:
        image: The input image as a NumPy array.
        width: The desired width (optional).
        height: The desired height (optional).

    Returns:
        The resized image as a NumPy array.
    """
    original_height, original_width = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is not None:
      ratio = width / original_width
      new_height = int(original_height * ratio)
      new_size = (width, new_height)
    else:
      ratio = height / original_height
      new_width = int(original_width * ratio)
      new_size = (new_width, height)

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image



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
        #height_res = opencv_image.shape[0]//4
        #width_res = opencv_image.shape[1]//4   
        #opencv_image_resz = cv2.resize(opencv_image, (int(width_res), int(height_res)))
        opencv_image_resz = resize_image(opencv_image, width=1000, height=1000)	    
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
    start_pnt = (x-w//2,y-h//2)
    end_pnt = (x+w//2, y+h//2)
    txt_start_pnt = (x-w//2, y-h//2-5)

    color = (255, 0, 0)   
        
    img = cv2.rectangle(img, start_pnt, end_pnt, color, 10)
    img = cv2.putText(img, cl, txt_start_pnt, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)	
    return img
    	
    


def predict(model, url):
    return model.predict(url, confidence=50, overlap=70).json()
    #return model.predict(url, hosted=True).json()



def predict_def_loc(model, roi):
    results = model.predict(roi, confidence=10, overlap=70).json()
    if len(results['predictions']) == 0:
        return roi
    else:
        for i in range(len(results['predictions'])):
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
            roi = drawBoundingBox(roi, x, y, w, h, cl, cnf)
        return roi
    
	
	
def main():
    st.title('Defect Detection v2')

    # Roi Extraction model
    rf = Roboflow(api_key="0N1hjNKtBabtHfuP93Q8")
    project = rf.workspace("verify-gn-hnnai").project("herminio-object-detection")
    model = project.version(1).model

    # Defect Localization model
    rf = Roboflow(api_key="0N1hjNKtBabtHfuP93Q8")
    project = rf.workspace("verify-gn-hnnai").project("herminio-defect-localization")
    model_def_loc = project.version(1).model
    
                
    
                
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
            roi_list = []
            roi_res_list = []
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
                try:
                    margin = 0
                    roi = svd_img[y1-margin:y2+margin, x1-margin:x2+margin, :]
                    #roi = cv2.resize(roi, (640, 640))			
                    cv2.imwrite(f"roi_{str(roi_count)}.jpg", roi)
                    roi_list.append(roi)
                    roi_res = predict_def_loc(model_def_loc, roi)
                    roi_res_list.append(roi_res)
                except:
                    margin = 0
                    roi = svd_img[y1-margin:y2+margin, x1-margin:x2+margin, :]
                    #roi = cv2.resize(roi, (640, 640))			
                    cv2.imwrite(f"roi_{str(roi_count)}.jpg", roi)
                    roi_list.append(roi)
                    roi_res = predict_def_loc(model_def_loc, roi)
                    roi_res_list.append(roi_res)	
                


                #svd_img = drawBoundingBox(svd_img,x, y, w, h, cl, cnf)

            st.write('DETECTION RESULTS')    
            st.image(svd_img, caption='Resulting Image')
            for i in range(roi_count):
                st.image(roi_res_list[i], caption=f'roi_{str(i)}')
           

if __name__ == '__main__':
    main()
