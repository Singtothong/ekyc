# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:54:34 2021

@author: Cromagnon-PC
"""
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from deepface import DeepFace
from deepface.basemodels import VGGFace,FbDeepFace, ArcFace
from deepface.commons import functions
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time
import cv2

# load model
pretrained_models = {}

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Facial Recognition')

def main():
    id_uploaded = st.file_uploader("Thai National ID Card", type=["png","jpg","jpeg"])
    if id_uploaded is not None:
        image_id = np.asarray(bytearray(id_uploaded .read()), dtype=np.uint8)
        image_id  = cv2.imdecode(image_id , cv2.IMREAD_COLOR)
        st.image(image_id, caption='Thai National ID Card', use_column_width=True, channels="BGR")
        person_uploaded = st.file_uploader("Selfie Photo holding Thai National ID Card", type=["png", "jpg", "jpeg"])
        if person_uploaded is not None:
            image_person = np.asarray(bytearray(person_uploaded.read()), dtype=np.uint8)
            image_person = cv2.imdecode(image_person, cv2.IMREAD_COLOR)
            st.image(image_person, caption='Selfie Photo', use_column_width=True, channels="BGR")

    class_btn = st.button("Classify")
    if class_btn:
        if (id_uploaded is None) & (person_uploaded is None):
            st.write("Invalid command, please upload an image")
        else:

            with st.spinner('Model working....'):
                id_face = getfaceHaarcascades2(image_id)
                person_face = getfaceHaarcascades2(image_person)
                models = ['VGG-Face', 'DeepFace', 'ArcFace']
                metrics = ['cosine', 'euclidean', 'euclidean_l2']
                pretrained_models["VGG-Face"] = VGGFace.loadModel()
                pretrained_models["DeepFace"] = FbDeepFace.loadModel()
                pretrained_models["ArcFace"] = ArcFace.loadModel()
                verify = faceComparison(id_face, person_face, models, metrics, pretrained_models)
                similar = sum(verify) / len(verify)
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                ax.set_title("Thai National ID Card")
                ax.imshow(id_face[:, :, ::-1])
                ax = fig.add_subplot(1, 2, 2)
                ax.set_title("Selfie Photo")
                ax.imshow(person_face[:, :, ::-1])
                time.sleep(1)
                st.pyplot(fig)
                if sum(verify) >= 5:
                    st.success('Pass')
                    st.success('The similarity between two images is {:.2f}%.'.format(similar*100))
                else:
                    st.error('Not pass')
                    st.error('The similarity between two images is {:.2f}%.'.format(similar*100))

def getfaceHaarcascades2(img_path):
    image = img_path
    gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if len(faces) >= 2:
        w_list = [w for (x, y, w, h) in faces]
        i = w_list.index(max(w_list))
        for (x, y, w, h) in faces[[i]]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_face = image[y:y + h, x:x + w]
            img_face = cv2.resize(img_face, (100, 100))

    else:
        i = 0
        (x, y, w, h) = faces[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_face = image[y:y + h, x:x + w]
        img_face = cv2.resize(img_face, (100, 100))
    return img_face

def faceComparison(image1, image2, models, metrics, pretrained_models):
    verify = []

    for model in models:
        for metric in metrics:
            if model == 'ArcFace':
                resp_obj = DeepFace.verify(image1, image2
                                           , model_name=model
                                           , model=pretrained_models[model]
                                           , distance_metric=metric
                                           , detector_backend='retinaface'
                                           , enforce_detection=False)
            elif model == 'OpenFace' and metric == 'euclidean':  # this returns same with openface euclidean l2
                continue
            else:
                resp_obj = DeepFace.verify(image1, image2
                                           , model_name=model
                                           , model=pretrained_models[model]
                                           , distance_metric=metric
                                           , enforce_detection=False)
            verify.append(resp_obj['verified'])
    return verify

if __name__ == "__main__":
    main()

        
        
        
