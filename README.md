# Computer Pointer Controller

The Computer Pointer Controller project uses the direction of the eyes in a gaze to move the mouse. To achieve this, four models are used together and these are comouter vision models. The models used are the face detection model, the landmark detection model, the headpose estimation model and gaze estimation model. 
The models produce some coordinates which is used to move the mouse pointer and the coordinates represent the direction of the eye gaze.


## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

First step will be to install OpenVINO

Then clone or download this repo.

Run the command pip3 install requirements.txt to install the required packages. 

Download the following models using the OpenVINO model downloader:

a. Face detection model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

b. Landmark regression model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

c. Head-pose estimation model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

d. Gaze estimation model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"


## Demo
To run a demo of this model, first launch the terminal. Once that has been launched, go to the src folder in the cloned directory and then use the command below to run it.

python3 main.py -i ../bin/demo.mp4 \
-mfd <this is the path to face detection model xml file> \
-mld <this is the path to landmark detection model xml file> \
-mhp <this is the path to head-pose estimation model xml file> \
-mgd <this is the path to gaze estimation model xml file>

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.
The required command line arguments are:

-inp, this is the path to the input which is the webcam or a video

-mfd, this is the path to the face detection model

-mld, this is the path to the landmark detection model

-mhp, this is the path to the head-pose estimation model

-mgd, this is the path to the gaze estimation model

The optional command line arguments are:

-lay, this is the path for MKLDNN (CPU)-targeted custom layers

-dev, target device type could be CPU, FPGA

-perf, this is the path that stores the statistics of performance i.e. inference time, frames per second, and model loading time.

-vf, specify flags from mfd, mld, mhp, mgd e.g. -vf mfd mld mhp mgd (seperate each flag by space) so the output of the models can be visualised

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results

The face detection model across all the precision model types has the most latency. This means that the inference speed of the four models combined is highly dependent on the inference speed of the face detection model.
Also, as the precision increases, the frames processed per second inccreases and we see that as the precision increases there is an increase in the computational resources required.

Looking at the three precisions, we see that FP32 has the highest accuracy. This is evident in the gaze estimation output. The gaze estimation model is the last to be utilised in the project and as such the precision losses are accumulated right from the first model to the last model (the gaze estimation model).


## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases

Multiple people in the video frame or the webcam frame might cause disruptions in the flow of inference if neither of the faces can be seen clearly. However, if the faces are detectible, then the first detected face will be used. 

In the event of poor ligtning, the project will log a message indicating that the face cannot be detected as a result of poor lights. There will also be an indication when the eyes cannot be seen even if the face is detected.