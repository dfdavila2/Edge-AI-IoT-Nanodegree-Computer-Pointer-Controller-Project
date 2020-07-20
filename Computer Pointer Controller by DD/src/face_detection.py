import os
import cv2
from openvino.inference_engine import IECore
from util_function import  preprocess_input

class Model_face_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU'):
        '''
        Set instance variables.
        '''
        self.device = device
        self.model = model_name
        self.model_structure = model_name
        self.model_weights = os.path.splitext(self.model_structure)[0] + ".bin"
        
        try:
            self.model = IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        
    def load_model(self):
        '''
        Load the model to the specified device.
        '''
        # Initialize the plugin
        core = IECore()
        
        # Load the network into the plugin
        self.exec_network = core.load_network(network=self.model, device_name=self.device)
        
        # Return the loaded inference plugin
        return self.exec_network

    def predict(self, image, threshold, display):
        '''
        Run predictions on the input image.
        '''        
        # Pre-process the image
        p_frame = preprocess_input(image, self.input_shape)
        
        # Start an asynchronous request
        self.exec_network.start_async(request_id=0, inputs={self.input_name: p_frame})
        
        # Wait for the result
        if self.exec_network.requests[0].wait(-1) == 0:
            
            # Get the results of the inference request
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            
            # Get the output images and the face coordinates
            out_image, face, face_coords = self.preprocess_output(image, outputs, threshold, display)
             
        return out_image, face, face_coords

    def preprocess_output(self, image, outputs, threshold, display):
        '''
        Preprocess the output before feeding it to the next model.
        '''
        # Grab the shape of the image
        width, height = image.shape[1], image.shape[0]
           
        face_coords = []
        face = image
        for box in outputs[0][0]:
            conf = box[2]
        
            # Check if confidence is bigger than the threshold
            if conf >= threshold:
                
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                
                face_coords.append(xmin)
                face_coords.append(ymin)
                face_coords.append(xmax)
                face_coords.append(ymax)
                
                face = image[ymin:ymax, xmin:xmax]
                
                if(display):
                    # Draw box
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)   
        
        return image, face, face_coords