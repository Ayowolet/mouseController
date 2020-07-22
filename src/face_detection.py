'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

#import numpy as np
from openvino.inference_engine import IENetwork
import cv2
import logging 
from models import Model 

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Face_Detection:
    '''
    Class for the Face Detection Model.
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
         # preprocess the image
        self.logger.info("Commence prepocessing of inputs and start inferencing")
        
        p_image = self.preprocess_input(image)
        # start asynchronous inference for specified request
        self.net.infer({self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            self.logger.info("Waiting for output of inference")
            outputs=self.net.requests[0].outputs[self.output_name]

            # select coords based on confidence threshold
            coordinates = self.preprocess_output(outputs)
            
            self.logger.info("The cropped face: {0}".format(coordinates))
            #return self.crop_output(coordinates,image)
            
            height = image.shape[0]
            width = image.shape[1]
        
            for x1, y1, x2, y2 in coordinates:
                xmin = int(x1 * width)
                ymin = int(y1 * height)
                xmax = int(x2 * width)
                ymax = int(y2 * height)
                image = image[ymin:ymax,xmin:xmax]
        
            return image
        

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
        coordinates = []
        for box in outputs[0][0]:
            conf = box[2]
            if conf > 0.6:
                coordinates.append(box[3:])
        return coordinates
        #raise NotImplementedError
