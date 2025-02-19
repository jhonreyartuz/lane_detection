from inference_worker.workflow import Model as BaseModel
from inference_worker.workflow import InferenceFailure, InferenceOutput
import cv2
import numpy as np
import torch


class Model(BaseModel):

    def __init__(self, enet_model, *, fileManager):
        super().__init__(fileManager=fileManager)
        self.enet_model = enet_model

    # Define a function to process and visualize the output
    def infer(self, input_image_path,  *args, **kwargs):

        result_image_path = self._fileManager.create_empty_file(file_extension=".png").name
        print("input_image_path", input_image_path[0])

        # Load and preprocess the input image
        input_image = cv2.imread(input_image_path[0])
        input_image = cv2.resize(input_image, (512, 256))  # Resize to the model's input size
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        input_image = input_image[..., None]
        input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1)  # Convert to tensor

        # Pass the input image through the model
        with torch.no_grad():
            binary_logits, instance_logits = self.enet_model(input_tensor.unsqueeze(0))



        # Post-process the model's output
        binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy()

        binary_seg_uint8 = (binary_seg * 225).astype(np.uint8)
        result = cv2.imwrite(result_image_path, binary_seg_uint8)

        print("result_image_path", result_image_path)
        instance_seg = torch.argmax(instance_logits, dim=1).squeeze().numpy()
        
        print("binary_seg", str(result))

        return InferenceOutput(outputFilePath=result_image_path, metadata={})
            
