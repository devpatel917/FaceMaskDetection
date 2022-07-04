import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image

array = []
def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to classify')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        array.append(image_data)

        img1 = Image.open(uploaded_file)
        img1 = img1.save("sample.png")

        st.image(image_data)
        
        

# # predict button -> output will be the uploaded image
# def predict():
def predict():
    model = load_model('model-001.model', compile=False)
    face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    labels_dict={0:'MASK',1:'NO MASK'}
    color_dict={0:(0,255,0),1:(0,0,255)}
    img = cv2.imread("sample.png")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    st.image(img)
    


def main():
    st.title('Mask Detection')
    load_image()
    result = st.button('Predict Image')
    if result:
        predict()
        # # st.image(array[0])
        # st.image(array[0])



    

if __name__ == '__main__':
    main()