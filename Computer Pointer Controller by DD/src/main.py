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
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD.")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")

    parser.add_argument("-o", "--output_dir", help = "Path to the output directory", type = str, default = None)
    
    parser.add_argument("-p", "--mouse_precision", required=False, default='high', type=str,
                        help="Set the precision for mouse movement: high, low, medium.")
                        
    parser.add_argument("-sp", "--mouse_speed", required=False, default='fast', type=str,
                        help="Set the speed for mouse movement: fast, slow, medium.")

    parser.add_argument("-m", "--mode", help = "async or sync mode", type = str, default = 'async')
                        
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
        log.error('Please enter a valid input! .jpg, .png, .bmp, .mp4, CAM')
        sys.exit()    
    return input_type

def infer_on_stream(args):
    """
    Initialize the inference networks, stream video to network,
    and output stats, video and control the mouse pointer.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    
    # Initialise the classes
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
    
    # Release the capture
    feed.close()
    # Destroy any OpenCV windows
    cv2.destroyAllWindows
 
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()

    '''
    if args.output_dir:
        with open(os.path.join(args.output_dir, 'stats.txt'), 'a') as f:
            f.write(str(round(model_load_time))+'\n')
    '''
    
    #Perform inference on the input stream
    infer_on_stream(args)

    if args.input == 'cam':
       input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "The input file does not exist"

    cap = cv2.VideoCapture(input_stream)
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(os.path.join(args.output_dir, "output.mp4"), 
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (initial_w, initial_h), True)
    
    frame_count = 0

    job_id = 1

    infer_time_start = time.time()

    if input_stream:
        cap.open(args.input)
        # Adjust DELAY to match the number of FPS of the video file

    if not cap.isOpened():
        logger.error("ERROR! Unable to open video source")
        return

    if args.mode == 'sync':
        async_mode = False
    else:
        async_mode = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print ("Error: BREAKING")
            break

        frame_count += 1
        looking = 0
        POSE_CHECKED = False

        if frame is None:
            log.error("ERROR: FRAME is nowhere to be found")
            break

        initial_w = int(cap.get(3))
        initial_h = int(cap.get(4))

        # Start asynchronous inference for specified request
        inf_start_fd = time.time()
        
        # Results of the output layer of the network
        #coords, frame = face_detection_network.predict(frame)
        
        det_time_fd = time.time() - inf_start_fd

        # Draw performance stats
        inf_time_message = "Face Inference time: {:.3f} ms.".format(det_time_fd * 1000)
        #
        if POSE_CHECKED:
            cv2.putText(frame, "Head pose Inference time: {:.3f} ms.".format(det_time_hp * 1000), (0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, inf_time_message, (0, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 0, 0), 1)
        out.write(frame)
        if frame_count%10 == 0:
            print("Inference time = ", int(time.time()-infer_time_start))
            print('Frame count {} and video length {}'.format( frame_count, video_len))
        if args.output_dir:
            total_time = time.time() - infer_time_start
            with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
                f.write(str(round(total_time, 1))+'\n')
                f.write(str(frame_count)+'\n')

    if args.output_dir:
        with open(os.path.join(args.output_dir, 'stats.txt'), 'a') as f:
            f.write(str(round(model_load_time))+'\n')

    # Clean all models
    face_detection_network.clean()
    pose_det.clean()
    land_det.clean()
    gaze_est.clean()
    # release cv2 cap
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    sys.exit()
