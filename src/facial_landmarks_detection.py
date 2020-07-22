'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork
import cv2
import logging
from models import Model 

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Facial_Landmarks_Detection:
    '''
    Class for the Facial Landmarks Detection Model.
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
        # synchronous inference
        self.net.infer({self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            outputs=self.net.requests[0].outputs[self.output_name]

            # select coords based on confidence threshold
            self.logger.info("Obtain coordinatess of the left and right eyes")
            coords = self.preprocess_output(outputs)
            self.logger.info("Return coordinates and the cropped Face")
            height = image.shape[0]
            width = image.shape[1]
        
            l_x0 = int(coords[0]*width) - 10
            l_x1 = int(coords[0]*width) + 10
            l_y0 = int(coords[1]* height) - 10
            l_y1 = int(coords[1]*height) + 10
            r_x0 = int(coords[2]*width) - 10
            r_x1 = int(coords[2]*width) + 10
            r_y0 = int(coords[3]*height) - 10
            r_y1 = int(coords[3]*height) + 10
        
        
            l_eye = image[l_y0:l_y1,l_x0:l_x1]
            r_eye = image[r_y0:r_y1,r_x0:r_x1]

            cv2.rectangle(image, (l_x0, l_y0), (l_x1, l_y1), (0, 0, 255), 2)
            cv2.rectangle(image, (r_x0, r_y0), (r_x1, r_y1), (0, 0, 255), 2)

            cv2.imwrite("FacialLandmark.jpg", image)

            return l_eye,r_eye, image
            

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_size = (self.input_shape[3], self.input_shape[2])
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
        outputs = outputs[0]
        self.logger.info("Here are the left and right eye coordinates: {0}".format(outputs))
        xl=outputs[0][0]
        yl=outputs[1][0]
        xr=outputs[2][0]
        yr=outputs[3][0]
        return (xl,yl,xr,yr)
    
        #raise NotImplementedError
