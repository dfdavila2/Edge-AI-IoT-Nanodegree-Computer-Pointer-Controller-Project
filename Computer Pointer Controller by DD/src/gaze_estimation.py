
import os
import cv2
import math
from openvino.inference_engine import IECore
from util_function import  preprocess_input

class Model_gaze_estimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
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

        self.input_name=[i for i in self.model.inputs.keys()]
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=[o for o in self.model.outputs.keys()]   

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

    def predict(self, image, left_eye, right_eye, eyes_center, head_pose_angles, display):
        '''
        Run predictions on the input image.
        '''
        # Pre-process the images
        p_left_eye = preprocess_input(left_eye, self.input_shape)
        p_right_eye = preprocess_input(right_eye, self.input_shape)
        
        # Start an asynchronous request #
        self.exec_network.start_async(request_id=0, inputs={'left_eye_image': p_left_eye,
                                                         'right_eye_image': p_right_eye,
                                                         'head_pose_angles': head_pose_angles})
        
        # Wait for the result
        if self.exec_network.requests[0].wait(-1) == 0:
            
            # Get the results of the inference request
            outputs = self.exec_network.requests[0].outputs[self.output_name[0]]
            
            # Get the output image and the gaze vector
            out_image, gaze_vector = self.preprocess_output(image, outputs, eyes_center, display)
                    
        return out_image, gaze_vector

    def preprocess_output(self, image, outputs, eyes_center, display):
        '''
        Preprocess the output.
        '''
        gaze_vector =  outputs[0]
        
        #Draw output
        if(display):
        
            # Left eye center
            left_eye_center_x = int(eyes_center[0][0])
            left_eye_center_y = int(eyes_center[0][1])
            
            # Right eye center
            right_eye_center_x = int(eyes_center[1][0])
            right_eye_center_y = int(eyes_center[1][1])
            
            #Draw arrowed Lines
            cv2.arrowedLine(image, (left_eye_center_x, left_eye_center_y), (left_eye_center_x + int(gaze_vector[0] * 100), left_eye_center_y + int(-gaze_vector[1] * 100)), (255,0,0), 5)
            cv2.arrowedLine(image, (right_eye_center_x, right_eye_center_y), (right_eye_center_x + int(gaze_vector[0] * 100), right_eye_center_y + int(-gaze_vector[1] * 100)), (255,0,0), 5)

        return image, gaze_vector 