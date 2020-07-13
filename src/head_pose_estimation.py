
import os
import cv2
from math import cos, sin, pi
from openvino.inference_engine import IECore
from util_function import  preprocess_input

class Model_head_pose_estimation:
    '''
    Class for the Head Pose Estimation  Model.
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
        self.output_name= [i for i in self.model.outputs.keys()]
               
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

    def predict(self, image, face, face_coords, display):
        '''
        Run predictions on the input image.
        '''
        # Pre-process the image 
        p_frame = preprocess_input(face, self.input_shape)
        
        # Start an asynchronous request 
        self.exec_network.start_async(request_id=0, inputs={self.input_name: p_frame})
        
        # Wait for the result
        if self.exec_network.requests[0].wait(-1) == 0:
            
            # Get the results of the inference request
            outputs = self.exec_network.requests[0].outputs
            
            # Get the output image and the head pose angles
            out_image,  head_pose_angles  = self.preprocess_output(image, outputs, face_coords, display)
            
        return out_image,  head_pose_angles 
        
    def draw_outputs(self, image, head_pose_angles ,face_coords): 
        '''
        Draw model output on the image.
        '''
        # Head pose angles
        y=head_pose_angles[0]
        p=head_pose_angles[1]
        r=head_pose_angles[2]
        
        # Face coordinates
        xmin = face_coords[0]
        ymin = face_coords[1]      
        xmax = face_coords[2]
        ymax = face_coords[3] 
        
        # I took the below code from here:
        # https://sudonull.com/post/6484-Intel-OpenVINO-on-Raspberry-Pi-2018-harvest
        cos_r = cos(r * pi / 180)
        sin_r = sin(r * pi / 180)
        sin_y = sin(y * pi / 180)
        cos_y = cos(y * pi / 180)
        sin_p = sin(p * pi / 180)
        cos_p = cos(p * pi / 180)
        
        x = int((xmin + xmax) / 2)
        y = int((ymin + ymax) / 2)
        
        # Center to right
        cv2.line(image, (x,y), (x+int(70*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(70*cos_p*sin_r)), (0, 0, 255), thickness=3)
        # Center to top
        cv2.line(image, (x, y), (x+int(70*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(70*cos_p*cos_r)), (0, 255, 0), thickness=3)
        # Center to forward
        cv2.line(image, (x, y), (x + int(70*sin_y*cos_p), y + int(70*sin_p)), (255, 0, 0), thickness=3)
       
        return image

    def preprocess_output(self, image, outputs, face_coords, display):
        '''
        Preprocess the output before feeding it to the next model.
        '''
        y = outputs['angle_y_fc'][0][0]
        p = outputs['angle_p_fc'][0][0]
        r = outputs['angle_r_fc'][0][0]
        
        head_pose_angles  =  [y, p, r]
        
        if (display):
            out_image = self.draw_outputs(image,  head_pose_angles, face_coords)
        
        return out_image,  head_pose_angles