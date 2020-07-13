'''
This class can be used to feed input from an image, webcam, or video to your model.
'''
import cv2
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.cap=cv2.VideoCapture(self.input_file)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        first_batch = True
        while True:
            for i in range(10):
            
                 # If input_type is image return that image  
                if self.input_type=='image':
                    frame = self.cap
                    if first_batch:
                        flag = True 
                        first_batch = False                        
                    else:
                        flag= False
                     
                # If input_type is cam or video read the cap      
                else:
                    flag, frame=self.cap.read()
            yield flag, frame

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()
