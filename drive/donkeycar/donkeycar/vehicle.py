#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 10:44:24 2017

@author: wroscoe
"""

import time
import numpy as np
from threading import Thread
from .memory import Memory
from prettytable import PrettyTable
import traceback
import cv2

from donkeycar.parts.camera import PiCamera

class PartProfiler:
    def __init__(self):
        self.records = {}

    def profile_part(self, p):
        self.records[p] = { "times" : [] }

    def on_part_start(self, p):
        self.records[p]['times'].append(time.time())

    def on_part_finished(self, p):
        now = time.time()
        prev = self.records[p]['times'][-1]
        delta = now - prev
        thresh = 0.000001
        if delta < thresh or delta > 100000.0:
            delta = thresh
        self.records[p]['times'][-1] = delta

    def report(self):
        print("Part Profile Summary: (times in ms)")
        pt = PrettyTable()
        field_names = ["part", "max", "min", "avg"]
        pctile = [50, 90, 99, 99.9]
        pt.field_names = field_names + [str(p) + '%' for p in pctile]
        for p, val in self.records.items():
            # remove first and last entry because you there could be one-off
            # time spent in initialisations, and the latest diff could be
            # incomplete because of user keyboard interrupt
            arr = val['times'][1:-1]
            if len(arr) == 0:
                continue
            row = [p.__class__.__name__,
                   "%.2f" % (max(arr) * 1000),
                   "%.2f" % (min(arr) * 1000),
                   "%.2f" % (sum(arr) / len(arr) * 1000)]
            row += ["%.2f" % (np.percentile(arr, p) * 1000) for p in pctile]
            pt.add_row(row)
        print(pt)


class Vehicle:
    def __init__(self, mem=None):

        if not mem:
            mem = Memory()
        self.mem = mem
        self.parts = []
        self.on = True
        self.threads = []
        self.profiler = PartProfiler()
        # Added
        self.has2stop = False
        self.cumulativeThrottle = 0
        self.throttleCounter = 0
        self.reverse = False
        self.reverseCounter = 0
        self.previous = 0
        self.models = ['PK','STOP','YK','ARROW']
        self.tick = 3
        self.maxspeed = 0.71
        self.arrow = False
        self.arrow_sign = None
        self.arrow_sign_counter = 12
        self.resolution_switched = False
        self.cumulative_right = 0
        self.cumulative_left = 0
        self.started = False
        self.started_counter = 5

    def add(self, part, inputs=[], outputs=[],
            threaded=False, run_condition=None, add_beginning=False):
        """
        Method to add a part to the vehicle drive loop.

        Parameters
        ----------
            part: class
                donkey vehicle part has run() attribute
            inputs : list
                Channel names to get from memory.
            outputs : list
                Channel names to save to memory.
            threaded : boolean
                If a part should be run in a separate thread.
            run_condition : str
                If a part should be run or not
        """
        assert type(inputs) is list, "inputs is not a list: %r" % inputs
        assert type(outputs) is list, "outputs is not a list: %r" % outputs
        assert type(threaded) is bool, "threaded is not a boolean: %r" % threaded

        p = part
        print('Adding part {}.'.format(p.__class__.__name__))
        entry = {}
        entry['part'] = p
        entry['inputs'] = inputs
        entry['outputs'] = outputs
        entry['run_condition'] = run_condition

        if threaded:
            t = Thread(target=part.update, args=())
            t.daemon = True
            entry['thread'] = t

        if not add_beginning:
            self.parts.append(entry)
        else:
            self.parts = [entry] + self.parts
        self.profiler.profile_part(part)

    def remove(self, part):
        """
        remove part form list
        """
        self.parts.remove(part)

    def start(self, rate_hz=10, max_loop_count=None, verbose=False):
        """
        Start vehicle's main drive loop.

        This is the main thread of the vehicle. It starts all the new
        threads for the threaded parts then starts an infinite loop
        that runs each part and updates the memory.

        Parameters
        ----------

        rate_hz : int
            The max frequency that the drive loop should run. The actual
            frequency may be less than this if there are many blocking parts.
        max_loop_count : int
            Maximum number of loops the drive loop should execute. This is
            used for testing that all the parts of the vehicle work.
        verbose: bool
            If debug output should be printed into shell
        """

        try:

            self.on = True

            for entry in self.parts:
                if entry.get('thread'):
                    # start the update thread
                    entry.get('thread').start()

            # wait until the parts warm up.
            print('Starting vehicle at {} Hz'.format(rate_hz))

            loop_count = 0
            while self.on:
                start_time = time.time()
                loop_count += 1

                self.update_parts()

                # stop drive loop if loop_count exceeds max_loopcount
                if max_loop_count and loop_count > max_loop_count:
                    self.on = False

                sleep_time = 1.0 / rate_hz - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    # print a message when could not maintain loop rate.
                    if verbose:
                        print('WARN::Vehicle: jitter violation in vehicle loop '
                              'with {0:4.0f}ms'.format(abs(1000 * sleep_time)))

                if verbose and loop_count % 200 == 0:
                    self.profiler.report()

        except KeyboardInterrupt:
            pass
        except Exception as e:
            traceback.print_exc()
        finally:
            self.stop()

    def preProcessImage(self,inputs,keyword, arrow):
        if not arrow and keyword != "ARROW":
            inputs = cv2.resize(inputs, (160,120))
        if(keyword == "SD"):
            input_processed = inputs[60:,:,:]
        elif(keyword == "PK"):
            input_processed = inputs[60:90,80:,:]
        elif(keyword == "YK"):
            input_processed = inputs[60:,20:-20,:]
        elif(keyword == "STOP"):
            input_processed = inputs[10:-50,20:-20,:]
        elif(keyword == "ARROW"):
            #input_processed = inputs[360:-315,380:-380,:]
            input_processed = np.copy(inputs[365:-315,380:-380,:])
            #input_processed[:,:,0] = 0
            #input_processed[:,:,1] = 0
            #input_processed[input_processed[:,:,2] > 140] = 255
            #input_processed[:,:,2][input_processed[:,:,2] <= 140] = 0
            input_processed = cv2.cvtColor(input_processed, cv2.COLOR_RGB2HSV)
            max_chan = np.max(input_processed[:,:,1])
            input_processed[input_processed[:,:,1] > max_chan-40] = 255
            input_processed[input_processed[:,:,1] <= max_chan-40] = 0
            #input_processed = input_processed/255
        return(input_processed)

    def update_parts(self):
        '''
        loop over all parts
        '''
        for entry in self.parts:
            run = True
            model_type = 'other'
            # check run condition, if it exists
            if entry.get('run_condition'):
                run_condition = entry.get('run_condition')
                if run_condition[0:3] == "run":
                    model_type = run_condition.split("_")[-1]
                    if model_type != self.models[self.tick] and model_type not in ["SD"]: #Check if model needs to be run
                        continue
                    elif not self.started and model_type == "ARROW": #If we haven't started moving then do not run arrow sign
                        continue
                    elif model_type == "SD" and self.started and self.reverse: #Check if we need to reverse
                        outputs = (0,-1)
                        self.reverseCounter -= 1
                        if self.reverseCounter == 0:
                            self.reverse = False
                        self.mem.put(entry['outputs'], outputs)
                        continue
                    run_condition = "run_pilot"
                elif run_condition == "camera" and not self.resolution_switched: # Check if we detected arrow sign and if we did then change camera resolution
                    if self.arrow_sign:
                        self.arrow = True
                        inputs = self.mem.get(entry['inputs'])
                        self.remove(entry)
                        p = entry['part']
                        p.shutdown()
                        cam = PiCamera(image_w=160, image_h=120, image_d=3, framerate=20, vflip=False, hflip=False)
                        self.add(cam, inputs=[], outputs=['cam/image_array'], threaded=True, add_beginning=True)
                        outputs = cam.run(*inputs)
                        self.mem.put(entry['outputs'], outputs)
                        self.parts[0].get('thread').start()
                        self.resolution_switched = True
                        continue
                if run_condition != "camera":
                    run = self.mem.get([run_condition])[0]

            if run:
                # get part
                p = entry['part']
                # start timing part run
                self.profiler.on_part_start(p)
                # get inputs from memory
                inputs = self.mem.get(entry['inputs'])
                if model_type != "other":
                    inputs = inputs
                #If it is not model, then we run it always
                if model_type == 'other':
                    if entry.get('thread'):
                        outputs = p.run_threaded(*inputs)
                    else:
                        outputs = p.run(*inputs)
                elif model_type == "ARROW": #If current model is arrow
                    print("Arrow")
                    if self.arrow_sign == None: #If we havent detected arrow sign
                        inputs[0] = self.preProcessImage(inputs[0], model_type, self.arrow)
                        if entry.get('thread'):
                            outputs = p.run_threaded(*inputs)
                        else:
                            outputs = p.run(*inputs)
                        print(outputs)
                        self.cumulative_left += outputs[0]
                        self.cumulative_right += outputs[1]
                        #Check if cumulatitve probability for given direction
                        if self.cumulative_left >= 3.0:
                            self.arrow_sign = -0.7
                        elif self.cumulative_right >= 3.0:
                            self.arrow_sign = 0.7
                        #if outputs[0] > 0.7:
                        #    self.arrow_sign = -0.7
                        #elif outputs[1] > 0.7:
                        #    self.arrow_sign = 0.7
                    else: #If arrow sign was detected, then remove from parts
                        self.profiler.on_part_finished(p)
                        self.remove(entry)
                        continue
                    print(model_type)
                    print(outputs)
                else:
                    #print(self.has2stop)
                    #Run for every type exept for self_driving model if previous model has detect it has to stop
                    if not self.has2stop:
                    # run the part
                        print(model_type)
                        inputs[0] = self.preProcessImage(inputs[0], model_type, self.arrow)
                        if entry.get('thread'):
                            outputs = p.run_threaded(*inputs)
                        else:
                            outputs = p.run(*inputs)
                        if model_type == 'SD':
                            if self.arrow_sign != None:
                                self.tick = (self.tick+1)%3
                            self.previous = outputs[0]
                            if outputs[1] < 0.20 and self.started:
                                self.cumulativeThrottle += outputs[1]
                                self.throttleCounter += 1
                                if self.throttleCounter >= 12: #This value needs to be tuned
                                    self.reverse = True
                                    self.reverseCounter = 10 #This value needs to be tuned
                                    self.throttleCounter = 0
                            else:
                                if not self.started and outputs[1] > 0.2:
                                    if self.started_counter > 0:
                                        self.started_counter -= 1
                                    else:
                                        self.started = True
                                self.cumulativeThrottle = 0
                                self.throttleCounter = np.maximum(0,self.throttleCounter - 3)
                        else:
                            if outputs[1] > 0.5: #If extra model detected stopping condition
                                self.has2stop = True
                    # If the sd model was run run the following
                    else:
                        if model_type == 'SD':
                            #If we need to stop then set throttle to zero.
                            outputs = (self.previous,0)
                            # If the self.has2stop == True then set it to False for next frame.
                            self.has2stop = False
                    # save the output to memory

                if model_type != "other":
                        print(outputs)
                        #a = 0
                
                #Run only for "SD" model
                if outputs is not None and model_type not in ['PK', 'YK', 'STOP', 'ARROW']:
                    if model_type != "other":
                        #outputs = (outputs[0], -1*outputs[1])
                        if outputs[1] > 0.2:
                            #self.maxspeed += 0.0005
                            outputs = (outputs[0], 0.8)#np.minimum(outputs[1]*1.1, 0.825))
                            #outputs = (outputs[0],np.minimum(np.minimum(outputs[1],self.maxspeed),0.75))
                        if outputs[1] < -0.1: #If model predicts negative value (wants to reverse) then set throttle to -1.0
                            outputs = (0,-1.0)
                        if outputs[0] > 0.3: #If model turns right then increase throttle
                            outputs = (outputs[0],outputs[1]*1.05)
                        if self.arrow_sign != None and not self.resolution_switched:
                            outputs = (0,0)
                        elif self.arrow_sign != None and self.arrow_sign_counter > 0: #If arrow sign was detected then notch the car in given direction
                            if self.arrow_sign > 0:
                                outputs = (np.maximum(outputs[0],-1*outputs[0]) +0.1,outputs[1])
                            else:
                                outputs = (np.minimum(outputs[0],-1*outputs[0]) -0.1,outputs[1])
                            self.arrow_sign_counter -= 1
                    self.mem.put(entry['outputs'], outputs)
                # finish timing part run
                self.profiler.on_part_finished(p)

    def stop(self):
        print('Shutting down vehicle and its parts...')
        for entry in self.parts:
            try:
                entry['part'].shutdown()
            except AttributeError:
                # usually from missing shutdown method, which should be optional
                pass
            except Exception as e:
                print(e)

        self.profiler.report()
