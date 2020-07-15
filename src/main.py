'''
Program to implement the various pipelines necessary for the pointer movement direction
'''

import cv2
import argparse
import logging
from input_feeder import InputFeeder
from gaze_estimation import Gaze_Estimation
from mouse_controller import MouseController
from facial_landmarks_detection import Facial_Landmarks_Detection
from head_pose_estimation import Head_Pose_Estimation
from face_detection import Face_Detection
import time
import os

logging.basicConfig(filename='mouse_controller.log', level=logging.DEBUG)


CPU_EXTENSION="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

performance_directory_path="../"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    # Argument groups are added
    reqd = parser.add_argument_group('required arguments')
    option = parser.add_argument_group('optional arguments')

    # -- create the arguments
    
    option.add_argument("-mfd", default='../models/fp32/face-detection-adas-binary-0001', help="This contains the path to face detection model", required = False)
    option.add_argument("-mld", default='../models/fp32/landmarks-regression-retail-0009', help="This contains the path to facial landmarks detection model", required = False)
    option.add_argument("-mhp", default='../models/fp32/head-pose-estimation-adas-0001', help="This contains the path to head pose estimation detection model", required = False)
    option.add_argument("-mgd", default='../models/fp32/gaze-estimation-adas-0002', help="This contains the path to gaze detection model", required = False)
    
    option.add_argument("-lay", default = CPU_EXTENSION, help="MKLDNN (CPU)-targeted custom layers.", required=False)
    option.add_argument("-dev", default='CPU', help="Specify the target device type")
    reqd.add_argument("-inp", help="path to video/image file or 'cam' for webcam", required = True)
    option.add_argument("-perf", help="path to store performance stats", required = False)
    option.add_argument("-vf", nargs='+', help="specify flags from mfd, mld, mhp, mgd e.g. -vf mfd mld mhp mgd so the output of the models can be visualised. Ensure that each flag is separated by a space.", default=[], required=False)
    args = parser.parse_args()

    return args

def model_pipelines(args):
    
    # Parameters which were parsed are assigned
    
    device = args.dev
    customLayers = args.lay
    inputFile = args.inp
    visual_flag = args.vf
    faceDetectionModel = args.mfd
    landmarksDetectionModel = args.mld
    headPoseEstimationModel = args.mhp
    gazeDetectionModel = args.mgd
    
    # Logging is enabled 
    log = logging.getLogger(__name__)

    # The feed is initialised
    single_image = ['jpg','tif','png','jpeg', 'bmp']
    if inputFile.split(".")[-1].lower() in single_image:
        input_feed = InputFeeder('image',inputFile)
    elif args.inp == 'cam':
        input_feed = InputFeeder('cam')
    else:
        input_feed = InputFeeder('video',inputFile)

    # Feed data is loaded
    input_feed.load_data()

    # The models are initialised and loaded here

    face_model_load_start_time = time.time()
    pipeline_faceDetection = Face_Detection(faceDetectionModel, device, customLayers)
    pipeline_faceDetection.load_model()
    face_model_load_time = time.time() - face_model_load_start_time
    
    landmark_model_load_start_time = time.time()
    pipeline_landmarksDetection = Facial_Landmarks_Detection(landmarksDetectionModel, device, customLayers)
    pipeline_landmarksDetection.load_model()
    landmark_model_load_time = time.time() - landmark_model_load_start_time
    
    headpose_model_load_start_time = time.time()
    pipeline_headPoseEstimation = Head_Pose_Estimation(headPoseEstimationModel, device, customLayers)
    pipeline_headPoseEstimation.load_model()
    headpose_model_load_time = time.time() - headpose_model_load_start_time
    
    gaze_model_load_start_time = time.time()
    pipeline_gazeDetection = Gaze_Estimation(gazeDetectionModel, device, customLayers)
    pipeline_gazeDetection.load_model()
    gaze_model_load_time = time.time() - gaze_model_load_start_time
    
    load_time = [face_model_load_time, landmark_model_load_time, headpose_model_load_time, gaze_model_load_time]
    
    
    # count the number of frames
    frameCount = 0

    # collate frames from the feeder and feed into the detection pipelines
    for _, frame in input_feed.next_batch():

        if not _:
            break
        frameCount += 1
        
        if frameCount % 5 == 0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))

        key = cv2.waitKey(60)
        face_inference_start_time = time.time()
        face_crop = pipeline_faceDetection.predict(frame)
        face_inference_time = time.time() - face_inference_start_time

        if 'mfd' in visual_flag:
            
            cv2.imshow('The cropped face', face_crop)

        if type(face_crop) == int:
            log.info("No face can be detected")
            
            if key == 27:
                break
            
            continue
       
        landmark_inference_start_time = time.time()
        eye_image_left, eye_image_right, face_landmarked = pipeline_landmarksDetection.predict(face_crop.copy())
        landmark_inference_time = time.time() - landmark_inference_start_time    

        if eye_image_left.any() == None or eye_image_right.any() == None:
            log.info("Landmarks could not be detected, check that the eyes are visible and the image is bright")
            continue
        
        if 'mld' in visual_flag:
            cv2.imshow('Face output', face_landmarked)
                
        headpose_inference_start_time = time.time()
        head_pose_angles, himage = pipeline_headPoseEstimation.predict(face_crop.copy())   
        headpose_inference_time = time.time() - headpose_inference_start_time

        if 'mhp' in visual_flag:
            cv2.imshow('Head Pose Angles', himage)
        
        gaze_inference_start_time = time.time()
        x,y = pipeline_gazeDetection.predict(eye_image_left ,eye_image_right, head_pose_angles)
        gaze_inference_time = time.time() - gaze_inference_start_time
        
        if 'mgd' in visual_flag:
            cv2.putText(face_landmarked, "Estimated x:{:.2f} | Estimated y:{:.2f}".format(x,y), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0,255,0),1)
            cv2.imshow('Gaze Estimation', face_landmarked)

        mouseVector = MouseController('medium','fast')
        
        
        if frameCount % 5 == 0:
            mouseVector.move(x,y)
          

        if key == 27:
            break
        
        inference_time = [face_inference_time, landmark_inference_time, headpose_inference_time, gaze_inference_time]
        results(args, inference_time, load_time)
        
    log.info("End of this run")
    cv2.destroyAllWindows()
    input_feed.close()
        

def results(args, inference_time, load_time):
    
    face_inference_time = inference_time[0]
    landmark_inference_time = inference_time[1]
    headpose_inference_time = inference_time[2]
    gaze_inference_time = inference_time[3]
    
    face_model_load_time = load_time[0]
    landmark_model_load_time = load_time[1]
    headpose_model_load_time = load_time[2]
    gaze_model_load_time = load_time[3]
    
    if args.perf != None and inference_time[0] != 0 and inference_time[1] != 0 and inference_time[2] != 0 and inference_time[3] != 0:
        output_path = performance_directory_path + args.perf
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    face_fps = 1 / face_inference_time
    landmark_fps = 1 / landmark_inference_time
    headpose_fps = 1 / headpose_inference_time
    gaze_fps = 1 / gaze_inference_time

    with open(os.path.join(output_path, 'face_model_stats.txt'), 'w') as file:
        file.write(str(face_inference_time)+'\n')
        file.write(str(face_fps)+'\n')
        file.write(str(face_model_load_time)+'\n')
           
    with open(os.path.join(output_path, 'landmark_model_stats.txt'), 'w') as file:
        file.write(str(landmark_inference_time)+'\n')
        file.write(str(landmark_fps)+'\n')
        file.write(str(landmark_model_load_time)+'\n')

    with open(os.path.join(output_path, 'headpose_model_stats.txt'), 'w') as file:
        file.write(str(headpose_inference_time)+'\n')
        file.write(str(headpose_fps)+'\n')
        file.write(str(headpose_model_load_time)+'\n')

    with open(os.path.join(output_path, 'gaze_model_stats.txt'), 'w') as file:
        file.write(str(gaze_inference_time)+'\n')
        file.write(str(gaze_fps)+'\n')
        file.write(str(gaze_model_load_time)+'\n')          
    
    return None
    
    

def main():
    args = get_args()
    model_pipelines(args) 

if __name__ == '__main__':
    main()