import cv2
from torchvision import transforms
import streamlit as st
from PIL import Image
import numpy as np
import torch
# from model import saved_model
import os
import time
import torch.nn as nn
import torch

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(784, 1),
    )
  def forward(self, x):
    return self.model(x)


def saved_model():
  model= Model()
  model.load_state_dict(torch.load('saved_modelcyclone', map_location=torch.device('cpu')))
  return model


# def provide_grey_scale(path,name):
#   image_path = os.path.join(path,name)
#   GSimage = Image.open(image_path)
#   time.sleep(5)
#   st.image(GSimage, caption="This is the grey scale image")
  
  
# Function to predict cyclone intensity using your ML model
def predict_intensity(image):
    # Assuming you have a function called predict_intensity_model
    # that takes an image array and returns the predicted intensity
    model = saved_model()
    # path = 'CYCLONE_DATASET_INFRARED/30.jpg'
    img = cv2.imread(image)
    img = np.array(img)
    totensor = transforms.ToTensor()
    img = totensor(img)
    resize = transforms.Resize(size=(250, 250))
    img = resize(img)
    img = torch.unsqueeze(img, 0)
    img = img.to("cpu")
    x =model(img)
    return x

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')
import numpy as np
# Setup the sidebar
with st.sidebar: 
    st.title('CycloNet Model')
    st.image('logo.png')
    st.info('This application is originally developed to estimate the intensity of a cyclone, resulting in harnessing the power of nature.')

st.title('A Full Stack App for Cyclone Intensity Estimation') 
# Generating a list of options or videos 
options = os.listdir('CYCLONE_DATASET_INFRARED')
options.sort()
selected_image = st.selectbox('Choose image', options)
#st.info(selected_video)
# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The selected image is displayed below in jpg format.')
        file_path = os.path.join('CYCLONE_DATASET_INFRARED', selected_image)
        #st.info(file_path)
        # os.system(f'ffmpeg -i {file_path} -vcodec libx264 test.mp4 -y')
        image = Image.open(file_path)
        # image = open(file_path) 
        # video_bytes = video.read() 
        st.image(image, caption="This is the uploaded image")
        
        # st.info("The respective Grey scale image:")
        # grey_scale_path = os.path.join("content\CYCLONE_DATASET_FINAL")
        # image_name = os.path.basename(file_path)
        # provide_grey_scale(grey_scale_path,image_name)
        


    with col2: 
        # st.info('This is all the machine learning model sees when making a prediction')
        img = cv2.imread(file_path)
        img = np.array(img)
        totensor = transforms.ToTensor()
        img = totensor(img)
        # st.info(img)
        # video, annotations = load_data(tf.convert_to_tensor(file_path))
        # st.info(type(video))
        # x=video.numpy()
        # st.info(type(x))
    
        # im = Image.fromarray(np.uint8(cm.gist_earth(x)*255))
        #imageio.mimsave('animation.gif',x , duration=10)
        # st.image('animation.gif', width=400) 

        #st.info('This is the output of the machine learning model as tokens')
        # model = load_model()
        # yhat = model.predict(tf.expand_dims(video, axis=0))
        # decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        #st.text(decoder)
        st.info("This is the tensor shape of the image uploaded:")
        st.text(img.shape)
        
        resize = transforms.Resize(size=(250, 250))
        img = resize(img)
        st.info("Shape after preprocessing:")
        st.text(img.shape)
        

        x=predict_intensity(file_path)
        # Convert prediction to text
        st.info('The tensor value of intensity')
        st.text(x)
        
        # st.info("This is the true value:(in knots per mile)")
        # intensity = image_name.split(".")[0]
        # st.text(intensity)
        
        st.info('The Predicted intensity value:(in knots per mile)')
        st.text(x.item())
        
        
        # converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        # st.text(converted_prediction)
        
