import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.applications import mobilenet_v2
from cv2 import cv2 

chan_dim = -1

if K.image_data_format() == "channels_first":
  input_shape = (depth, height, width)
  chan_dim = 1

def about():
	st.write(
		'''
		
	Katalog çekimleri bir kez yapıldığında bu resimlerin pazarlanması için bir çok ülke ve websitelerinde sergilenmesi gerekmektedir. 
    Her website için resim üzerinde farklı değişiklikler yapılma ihtiyacı doğmaktadır. 
    Ama elbise görüntüsü manken üzerinde olduğu için değişiklik yapabilme konusunda kısıtlamalar oluşmaktadır. 
    Bu geliştirilen proje bu sorunu çözebilme aşamasında kullanılabilecek bir proje olmaktadır. 
    Bu proje ile birlikte sadece elbise görüntüsü elde edilebilecektir. 
    Bu da görsel üzerinde kolaylıkla değişiklik yapılabilmesine olanak sağlayacaktır. 
    U-NET modeli kullanılarak girdi olarak verilen elbise görselinin maske görüntüsü elde edilmiştir. 
    Daha sonra bu maske görüntüsü kullanarak istenilen çıktı elde edilmiştir.
				
		''')


model = load_model('dress_segm.h5', compile=False)

def main():
  st.title("Arka Plan ve Manken Görüntüsü Olan Görsellerden Elbise Resminin İzole Edilmesi")

  activities = ["Uygulama", "Hakkında"]
  choice = st.sidebar.selectbox("Seçenekler:", activities)

  if choice == "Uygulama":
    image_file = st.file_uploader("Elbise Görselini Yükleyiniz...", type=['jpeg', 'png', 'jpg'])

    if image_file is not None:
      orig_image = Image.open(image_file)
      image = orig_image.resize((224,224))
      image = np.array(image)
      image = np.expand_dims(image, axis=0)
      st.image(image)
      
    if st.button("İzole Et!"):
      prediction = model.predict(image)
      prediction_image = prediction.reshape(224,224)
      pred = np.dstack([prediction_image, prediction_image, prediction_image])
      pred = (pred * 255).astype(np.uint8)
      cv2.imwrite('pred3.png', pred)
      original = Image.open(image_file)
      original = original.resize((224,224), Image.ANTIALIAS)
      original.save('original3.png')
      mask = cv2.imread('pred3.png')
      dress = cv2.imread('original3.png')
      _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
      # copy where we'll assign the new values
      background = np.copy(dress)
      # boolean indexing and assignment based on mask
      background[(mask==0).all(-1)] = [255,255,255]
      background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
      background = array_to_img(background)
      newsize = (224, 224)
      background = background.resize(newsize)
      st.image(background)

  elif choice == "Hakkında":
    about()

if __name__ == "__main__":
    main()
