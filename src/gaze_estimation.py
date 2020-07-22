'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import cv2
import logging
from models import Model 

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Gaze_Estimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    
    def __init__(self, model_name):
        Model.__init__(self, model_name) # initialize the base class
        # then create/override the variable
        #self.model_name = "faceDetectionModel"
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception:
            self.logger.exception("The Network could not be initialized. Check that you have the correct model path")
        
        self.load_model()
        Model.check_model(self, model_name)
        
    
    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core=IECore()
 
        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0:
            ### TODO: Add any necessary extensions ###
            if self.cpu_extension and "CPU" in self.device:
                self.core.add_extension(self.cpu_extension, self.device)
            else:
                self.logger.debug("Add CPU extension and device type or run layer with original framework")
                exit(1)
 
        self.net=self.core.load_network(network=self.model,device_name=self.device,num_requests=1)

        self.input_name=[i for i in self.model.inputs.keys()]
        self.input_shape=self.model.inputs[self.input_name[1]].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
 

    def predict(self, leftEye, rightEye, headPose):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.logger.info("Commence prepocessing of inputs and start inferencing")
        left_eye_image = self.preprocess_input(leftEye)
        right_eye_image = self.preprocess_input(rightEye)
        # start asynchronous inference for specified request
        self.logger.info("Commence sync request")
        self.net.infer({'head_pose_angles':headPose, 'left_eye_image': left_eye_image, 'right_eye_image': right_eye_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            
            outputs=self.net.requests[0].outputs[self.output_name]
            
            return self.preprocess_output(outputs)
       
    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_size=(self.input_shape[3], self.input_shape[2])
        image = cv2.resize(image,(image_size))
        image = image.transpose((2,0,1))
        image = image.reshape(1,*image.shape)

        return image
        #raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        x = outputs[0][0]
        y = outputs[0][1]

        return x,y
        #raise NotImplementedError
