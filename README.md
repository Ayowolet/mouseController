# Computer Pointer Controller

The Computer Pointer Controller project uses the direction of the eyes in a gaze to move the mouse. To achieve this, four models are used together and these are comouter vision models. The models used are the face detection model, the landmark detection model, the headpose estimation model and gaze estimation model. 
The models produce some coordinates which is used to move the mouse pointer and the coordinates represent the direction of the eye gaze.


## Project Set Up and Installation

To set up this project, the first step will be to install OpenVINO

Then clone or download this repo.

After downloading the repo, run the command **pip3 install requirements.txt** to install the required packages. 

Download the following models using the OpenVINO model downloader:
```
* Face detection model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

* Landmark regression model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

* Head-pose estimation model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

* Gaze estimation model

python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

## Demo
To run a demo of this model, first launch the terminal. Once that has been launched, go to the src folder in the cloned directory and then use the command below to run it.

To run the code using a demo video, use the command below
```
python3 main.py -inp ../bin/demo.mp4 
```
To run the code using a webcam, use the command below
```
python3 main.py -inp cam
```

## Documentation
The only required command line argument is:

```
-inp, this is where the user indicates the path to the input if it is a video or indicates if it is a webcam
```
The user can use other commands to modify the default settings of the program.
The optional command line arguments are:
```
-mfd, this sets the path to the face detection model
-mld, this sets the path to the landmark detection model
-mhp, this sets the path to the head-pose estimation model
-mgd, this sets the path to the gaze estimation model
-lay, this sets the path for targeted custom layers (MKLDNN)
-dev, target device type could be CPU, FPGA
-perf, this sets the path that stores the statistics of performance i.e. inference time, frames per second, and model loading time.
-vf, specify flags from mfd, mld, mhp, mgd e.g. -vf mfd mld mhp mgd so the output of the models can be visualised. Ensure that each flag is separated by a space.
```

## Directory
This is the structure of the project.

![Alt text](https://github.com/ayowolet/mouseController/blob/master/bin/tree.png)

## Benchmarks
The images below show the performance of the different models with the different precision types.

![Alt text](https://github.com/ayowolet/mouseController/blob/master/bin/inference_time.png)
![Alt text](https://github.com/ayowolet/mouseController/blob/master/bin/frames_per_second.png)
![Alt text](https://github.com/ayowolet/mouseController/blob/master/bin/model_load_time.png)

## Analysis of the above Benchmark Images

The face detection model across all the precision model types has the most latency. This means that the inference speed of the four models combined is highly dependent on the inference speed of the face detection model.
Also, as the precision increases, the frames processed per second inccreases and we see that as the precision increases there is an increase in the computational resources required.

Looking at the three precisions, we see that FP32 has the highest accuracy. This is evident in the gaze estimation output. The gaze estimation model is the last to be utilised in the project and as such the precision losses are accumulated right from the first model to the last model (the gaze estimation model).

### Edge Cases

Multiple people in the video frame or the webcam frame might cause disruptions in the flow of inference if neither of the faces can be seen clearly. However, if the faces are detectible, then the first detected face will be used. 

In the event of poor ligtning, the project will log a message indicating that the face cannot be detected as a result of poor lights. There will also be an indication when the eyes cannot be seen even if the face is detected.
