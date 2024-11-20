import torch
from Graph_preprocessing_functions import make_graph_for_image_slic, draw_graph, show_comparison_no_label, convert_to_data
from GNN_Data_cleanup import load_and_preprocess_images, load_and_preprocess_pred_images
import Utils as U
from GNN_Model import GNN
import HyperParameters
from skimage.segmentation import slic
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
import random
from PIL import Image
import requests
from io import BytesIO
import numpy as np

device = HyperParameters.device
amount = 10 #Number of images to run the model on (if doing the datasets pred data)
#Variable that decides whether to run the data on the datasets prediction data, or urls ('pred'/'url')
which = "pred" 
model = HyperParameters.MODEL_NAME + '.pth'

Model_0 = GNN(input_dim=HyperParameters.input_dim)
try:
    Model_0.load_state_dict(torch.load((U.MODEL_FOLDER / model).resolve()))
except:
    Model_0.load_state_dict(torch.load((U.MODEL_FOLDER / model).resolve(), map_location=torch.device('cpu')))
Model_0.to(device)

def model_on_prediction_data():
    #Run the model on the prediction folder data
    #preproccess the images then choose 'amount' randomly.
    num_correct = 0
    images = load_and_preprocess_pred_images(U.pred_folders)
    for x in range(amount):
        img_index = random.randint(0, len(images))
        img = images[img_index]
        segments = slic(img, n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
        show_comparison_no_label(img, segments)
        make_prediction(img)
        cor = input("Was I correct? (y/n): ")
        if cor == 'y':
            num_correct+=1
    
    print(f"Percentage correct: {num_correct/amount*100:2f}%")  

def model_on_new_images():
    #Take a given url, then download and preprocess it to run the model on
    while True:
        image_url = input("\nEnter image URL (q to quit): ")
        try:
            if image_url == 'q':
                break
            # Fetch the image from the URL
            response = requests.get(image_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Open the image using Pillow and convert to RGB
                img = Image.open(BytesIO(response.content)).convert("RGB").resize(HyperParameters.target_size)
                
                # Convert the image to a numpy array and normalize pixel values
                img_array = np.array(img, dtype=np.float32) / 255.0
                segments = slic(img_array, n_segments=HyperParameters.n_segments, sigma=HyperParameters.sigma)
                show_comparison_no_label(img_array, segments)
                make_prediction(img_array)

        except: 
            print("Invalid Image")

def make_prediction(img):
    #take given image, convert it into a graph
    img_graph = make_graph_for_image_slic(img)

    graph = from_networkx(img_graph)
    #Convert graph to data object
    data = convert_to_data(graph)
    data = data.to(device) #Send data to GPU if available
    print("Image converted to graph...")
    Model_0.eval()
    with torch.no_grad(): #Use the model's inference mode
        x = data.x
        edge_index = data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long)  # All nodes belong to the same graph, so all batch indices are 0
        batch = batch.to(device)
        prediction = F.softmax(Model_0.forward(x, edge_index, batch), dim=1) #Model's prediction
        predicted_class = prediction.argmax(dim=1) #Class prediction extracted from prediction 
        #Draw and label the graph with the model's prediction
        graph_label = f"\nPredicted class: {HyperParameters.CLASSES[predicted_class]}; Confidence: {prediction[0][predicted_class].item()*100:.2f}%"
        draw_graph(img_graph, graph_label)
        return predicted_class

if __name__ == "__main__":
    if which == 'pred':
        model_on_prediction_data()
    else:
        model_on_new_images()
