#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 02:02:45 2020

@author: ayowolet
"""

from openvino.inference_engine import IECore
import logging

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


class Model:
    
    
    def __init__(self, model_name, device='CPU', extensions=CPU_EXTENSION):
        self.model_name = model_name
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.cpu_extension = extensions
        self.logger = logging.getLogger(__name__)
    
    def layer_check(self):
        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network = self.model, device_name = self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0:
            ### TODO: Add any necessary extensions ###
            if self.cpu_extension and "CPU" in self.device:
                self.core.add_extension(self.cpu_extension, self.device)
            else:
                self.logger.debug("Add CPU extension and device type or run layer with original framework")
                exit(1)
        return

        
    def load_model(self, model_name, device = "CPU", cpu_extension = None):
        self.core = IECore()
        
        Model.layer_check(self)
        
        # load the model
        self.net = self.core.load_network(network = self.model, device_name = self.device, num_requests = 1)

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        
        return
    
    def check_model(self, model_Name):
        self.logger.info('Face_Detection Model Input shape: {0}'.format( str(self.input_shape) ))
        self.logger.info('Face_Detection Model Output shape: {0}'.format(str(self.output_shape)))

        Model.layer_check(self)
            
        return
        