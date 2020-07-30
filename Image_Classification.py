import streamlit as st 

# title 
html_temp=""" 
 <div style="background-color:royalblue;color:white; text-align:center; padding:15px;"><h1> Image Classification Using Transfer Learning</h1></div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

html_temp4=""" <br><br>
 <div style="background-color:royalblue;color:white; text-align:left; padding:15px;"><h4>What is Transfer Learning?</h4></div>
"""
st.markdown(html_temp4, unsafe_allow_html=True)
st.write("Transfer learning (TL) is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks.")

import torch
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

import train
import predict
import json
import pandas as pd
from bokeh.io import show, output_file
from bokeh.plotting import figure
import io


def main():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    checkpoint = torch.load('my_model.pt')

    arch = checkpoint['arch']
    num_labels = len(checkpoint['class_to_idx'])
    hidden_units = checkpoint['hidden_units']

    model = train.load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)
    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title("Testing Class Prediction")
    filename = st.file_uploader('Select the flower', encoding="auto", default="ftest.jpg")
    img= Image.open(filename)
    st.image(img,width=300, caption="Flower Selected")

    if st.button("Predict"):

        st.title("Predicted Results")
    # Process a PIL image for use in a PyTorch model
        def process_image(image):
            ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
                returns an Numpy array
            '''    
            img_loader = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor()])
            
            pil_image = Image.open(image)
            pil_image = img_loader(pil_image).float()
            
            np_image = np.array(pil_image)    
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
            np_image = np.transpose(np_image, (2, 0, 1))
                    
            return np_image


            def imshow(image, ax=None, title=None):
            #Imshow for Tensor
                if ax is None:
                     fig, ax = plt.subplots()
                
                # PyTorch tensors assume the color channel is the first dimension
                # but matplotlib assumes is the third dimension
                image = np.transpose(image, (1, 2, 0))
                
                # Undo preprocessing
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                
                # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
                image = np.clip(image, 0, 1)
                
                ax.imshow(image)
                
                return ax
        
            # Display an image along with the top 5 classes
        img =filename
        probs, classes = predict.predict(image=filename, checkpoint='my_model.pt', labels='cat_to_name.json', gpu=True)
        st.image(img)
        st.header("Table results of Possible Class")
        df = pd.DataFrame({'Classes':classes , 'Probability': probs})
        st.table(df)
        Classes =classes
        Probability = probs

        p = figure(x_range=Classes, plot_height=250, title="Possible Prediction",
           toolbar_location=None, tools="")

        p.vbar(x=Classes, top=Probability, width=0.6)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0

        st.bokeh_chart(p)
   
if __name__ == '__main__':
    main()
    