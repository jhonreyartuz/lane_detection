from inference_worker.workflow import ModelBuilder as BaseModelBuilder
import cv2
import numpy as np
from .lane_detector import ENet
import matplotlib.pyplot as plt
import torch
from .model import Model

class ModelBuilder(BaseModelBuilder):

    def build(self, model_file_paths, *args, **kwargs):
        enet_model = ENet(2, 4)  # Assuming you used the same model architecture

        # Load the trained model's weights
        print("model_file_paths",model_file_paths)
        enet_model.load_state_dict(torch.load(model_file_paths[0], map_location=torch.device('cpu')))
        enet_model.eval()  # Set the model to evaluation mode
        enet_model
        # print(enet_model)
        return Model(enet_model, fileManager=self._fileManager)

        

model_builder_class = ModelBuilder