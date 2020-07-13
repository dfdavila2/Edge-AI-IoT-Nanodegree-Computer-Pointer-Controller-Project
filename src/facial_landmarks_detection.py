
import os
import cv2
from openvino.inference_engine import IECore
from util_function import  preprocess_input

class Model_facial_landmarks_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device):
        '''
        Set instance variables.
        '''
        self.device=device
        self.model=model_name
        self.model_structure=model_name
        self.model_weights=os.path.splitext(self.model_structure)[0] + ".bin"
        
        try:
            self.model=IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        
    def load_model(self):
        '''
        Load the model to the specified device.
        '''     
        # Initialize the plugin
        core = IECore()
        
        # Load the network into the plugin
        self.exec_network = core.load_network(network=self.model, device_name=self.device)
        
        ### Return the loaded inference plugin ###
        return self.exec_network

    def predict(self, image, face, face_coords, display):
        '''
        Run predictions on the input image.
        '''
        ### Pre-process the image ###
        p_frame = preprocess_input(face, self.input_shape)
        
        ### Start an asynchronous request ###
        self.exec_network.start_async(request_id=0, inputs={self.input_name: p_frame})
        
        # Wait for the result
        if self.exec_network.requests[0].wait(-1) == 0:
            
            # Get the results of the inference request
            outputs = self.exec_network.requests[0].outputs[self.output_name]
        
            image, left_eye, right_eye, eyes_center = self.preprocess_output(outputs, face_coords, image, display)
        
        return image, left_eye, right_eye, eyes_center       
        
    def preprocess_output(self, outputs, face_coords, image, display):
        '''
        Preprocess the output before feeding it to the next model.
        '''
        # Get the landmarks from the outputs
        landmarks = outputs.reshape(1, 10)[0]

        # Grab the shape of the face
        # face_coords = (xmin,ymin,xmax,ymax)
        height = face_coords[3] - face_coords[1] #ymax-ymin
        width = face_coords[2] - face_coords[0]
        
        # Calculate coordinates for the left eye
        x_le = int(landmarks[0] * width) 
        y_le = int(landmarks[1]  *  height)
        
        #Consider offset of face from main image
        xmin_le = face_coords[0] + x_le - 30
        ymin_le = face_coords[1] + y_le - 30
        xmax_le = face_coords[0] + x_le + 30
        ymax_le = face_coords[1] + y_le + 30
         
        # Calculate coordinates for the right eye
        x_re = int(landmarks[2]  *  width)
        y_re = int(landmarks[3]  *  height)
        
        #Consider offset of face from main image
        xmin_re = face_coords[0] + x_re - 30
        ymin_re = face_coords[1] + y_re - 30
        xmax_re = face_coords[0] + x_re + 30
        ymax_re = face_coords[1] + y_re + 30
        
        if(display):
            # Draw the boxes 
            cv2.rectangle(image, (xmin_le, ymin_le), (xmax_le, ymax_le), (0,255,0), 3)        
            cv2.rectangle(image, (xmin_re, ymin_re), (xmax_re, ymax_re), (0,255,0), 3)
        
        # Eyes center
        left_eye_center =[face_coords[0] + x_le, face_coords[1] + y_le]
        right_eye_center = [face_coords[0] + x_re , face_coords[1] + y_re]      
        eyes_center = [left_eye_center, right_eye_center ]
        
        # Crop the left eye from the image
        left_eye = image[ymin_le:ymax_le, xmin_le:xmax_le]
        
        # Crop the right eye from the image
        right_eye = image[ymin_re:ymax_re, xmin_re:xmax_re]
        
        return image, left_eye, right_eye, eyes_center