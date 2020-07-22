'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork
import cv2
import logging
from models import Model 


CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Head_Pose_Estimation:
    '''
    Class for the Head Pose Estimation Model.
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
        
        Model.load_model(self, model_name)
        Model.check_model(self, model_name)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.logger.info("Commence prepocessing of inputs and start inferencing")
        
        p_image = self.preprocess_input(image)
        # sync inference
        self.logger.info("Inference result")
        outputs = self.net.infer({self.input_name: p_image})
        self.logger.info("Infered result")
        # wait for the result
        
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            outputs=self.net.requests[0].outputs
            
            return self.preprocess_output(outputs, image)


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_size = (self.input_shape[3], self.input_shape[2])
        image = cv2.resize(image,(image_size))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image
        #raise NotImplementedError

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        self.logger.info(" Getting the yaw, pitch, and roll angles ")
        angles = []
        
        angles.append(outputs['angle_y_fc'][0][0])
        angles.append(outputs['angle_p_fc'][0][0])
        angles.append(outputs['angle_r_fc'][0][0])
        
        cv2.putText(image, "Estimated yaw:{:.2f} | Estimated pitch:{:.2f}".format(angles[0],angles[1]), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0,255,0),1)
        cv2.putText(image, "Estimated roll:{:.2f}".format(angles[2]), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0,255,0),1)
        return angles, image
        #raise NotImplementedError
