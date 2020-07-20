import sys
import cv2
import time
import os
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder

from face_detection import Model_face_detection
from head_pose_estimation import Model_head_pose_estimation
from facial_landmarks_detection import Model_facial_landmarks_detection
from gaze_estimation import Model_gaze_estimation
from mouse_controller import MouseController

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--fd_model", required=True, type=str,
                        help="Path to an xml file of the Face Detection model.")
   
    parser.add_argument("-hp", "--hp_model", required=True, type=str,
                        help="Path to an xml file of the Head Pose Estimation model.")
                        
    parser.add_argument("-fl", "--fl_model", required=True, type=str,
                        help="Path to an xml file of the Facial Landmarks Detection model.")
                        
    parser.add_argument("-ge", "--ge_model", required=True, type=str,
                        help="Path to an xml file of the Gaze Estimation model.")
                       
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="CAM or path to image or video file.")
    
    parser.add_argument("-dis", "--display", required=False, default=True, type=str,
                        help="Flag to display the outputs of the intermediate models")

    parser.add_argument("-d", "--device", required=False, default="CPU", type=str,
                        help="Choose the target device used for inference: "
                             "CPU, IGPU, MYRIAD or FPGA.")
    
    parser.add_argument("-pt", "--prob_threshold",required=False, default=0.5, type=float,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    parser.add_argument("-o", "--output_dir", required=False, default = None, type = str,
                        help = "Path to the output directory"     )
    
    parser.add_argument("-p", "--mouse_precision", required=False, default='high', type=str,
                        help="Set the precision for mouse movement: high, low, medium.")
                        
    parser.add_argument("-sp", "--mouse_speed", required=False, default='fast', type=str,
                        help="Choose among 3 types of mouse speed: fast, slow, medium.")

    parser.add_argument("-m", "--mode", required=False, default = 'async', type = str,
                        help = "Select between sync or async mode")
                        
    return parser

def handle_input_type(input_stream):
    '''
     Handle image, video or webcam
    '''
    
    # Check if the input is an image
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        input_type = 'image'
        
    # Check if the input is a webcam
    elif input_stream == 'CAM':
        input_type = 'cam'
        
    # Check if the input is a video    
    elif input_stream.endswith('.mp4'):
        input_type = 'video'
    else: 
        log.warning('Please enter a valid input! .jpg, .png, .bmp, .mp4, CAM')
        sys.exit()    
    return input_type

def infer_on_stream(args):
    """
    Initialize the inference networks, stream video to network,
    and output stats, video and control the mouse pointer.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    
    # Initialising the classes
    face_detection_network = Model_face_detection(args.fd_model, args.device)
    head_pose_network = Model_head_pose_estimation(args.hp_model, args.device)
    facial_landmarks_network =  Model_facial_landmarks_detection(args.fl_model, args.device)
    gaze_estimation_network = Model_gaze_estimation(args.ge_model, args.device)
    
    MC = MouseController(args.mouse_precision, args.mouse_speed)
    
    start_load = time.time()
    
    # Load the models 
    face_detection_network.load_model()
    head_pose_network.load_model()
    facial_landmarks_network.load_model()
    gaze_estimation_network.load_model()
    
    print("All models have been loaded successfully...")
    
    end_load = time.time() -  start_load 
    
    # Handle the input stream
    input_type = handle_input_type(args.input)
    
    # Initialise the InputFeeder class
    feed = InputFeeder(input_type=input_type, input_file=args.input)
    
    # Load the video capture
    feed.load_data()
    
    start_inf = time.time()
    
    # Read from the video capture 
    for flag, frame in feed.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Run inference on the models     
        out_frame, face, face_coords = face_detection_network.predict(frame, args.prob_threshold, args.display)
        
        ## If no face detected move back to the top of the loop
        if len(face_coords) == 0:
            log.error("No face detected.")
            continue
            
        out_frame,  head_pose_angles = head_pose_network.predict(out_frame, face, face_coords, args.display)
        out_frame, left_eye, right_eye, eyes_center = facial_landmarks_network.predict(out_frame, face, face_coords, args.display)
        out_frame, gaze_vector = gaze_estimation_network.predict(out_frame, left_eye, right_eye, eyes_center, head_pose_angles, args.display)
        
        # Move the mouse
        MC.move(gaze_vector[0], gaze_vector[1])
        
        if key_pressed == 27:
            break
       
       # Display the resulting frame
        cv2.imshow('Visualization', cv2.resize(out_frame,(600,400)))
     
    end_inf = time.time() - start_inf
    
    print("Total model loading time: {} s\nTotal inference time: {} s".format(end_load, end_inf))
    
    print("Thanks for testing the app!\n")
    
    #Release the capture
    feed.close()
 
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()
    
    #Perform inference on the input stream
    infer_on_stream(args)

    if args.input == 'cam':
       input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "The input file does not exist"
    
    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)
    if (cap.isOpened() == False):
        log.error("Unable to read camera feed")
    else:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                  (frame_width, frame_height))
        org_width = int(cap.get(3))
        org_height = int(cap.get(4))

    frame_count = 0

    job_id = 1

    infer_time_start = time.time()

    if input_stream:
        cap.open(args.input)
        #Adjust DELAY to match the number of FPS of the video file

    if not cap.isOpened():
        logger.info("ERROR! Unable to open video source")
        return

    if args.mode == 'sync':
        async_mode = False
    else:
        async_mode = True

    #Release cv2 cap
    cap.release()

    #Destroy any OpenCV windows
    cv2.destroyAllWindows()

    #Release the video output file
    out.release()
           
if __name__ == '__main__':
    main()
    sys.exit()
