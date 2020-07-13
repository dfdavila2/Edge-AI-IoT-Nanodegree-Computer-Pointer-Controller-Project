import cv2

def preprocess_input(image, input_shape):
        '''
        Preprocess the data before feeding it into the model for inference.
        '''
        p_frame = cv2.resize(image, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        return p_frame