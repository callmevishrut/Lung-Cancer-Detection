# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:36:47 2018

@author: Vishrut Sharma
"""

import numpy as np
import pandas as pd
import matplotlib
import os

#import tensorflow as tf

#import cv2


import tkinter as tk
import tensorflow as tf
matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)

abouttxt = open("about.txt", 'r')
abttxt = abouttxt.read()

path_entry = 'F:/Minor Project 2/input'
patients = os.listdir(path_entry)
global patient_id
global x, y,IMG_SIZE_PX, SLICE_COUNT, keep_rate, n_classes, validation_data, prediction
global predict
patient_id = 1
global first_pass
first_pass = '...'
predict = 'benign'

#Back end code
def predict_result(pid):
    lung_data = np.load('lungdata-50-50-20.npy')
    predict_dataset = lung_data[pid]
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        #print([node.name for node in graph.as_graph_def().node])
        x = graph.get_tensor_by_name('Placeholder:0')
        y = graph.get_tensor_by_name('Placeholder_1:0')
        p = graph.get_tensor_by_name('MatMul_1:0')
        X = predict_dataset[0]
        Y = predict_dataset[1]
        pred = p.eval(feed_dict={x: X, y: Y})
        val = pred[0][0]
        print(val)
        if(val < 0):
            return "Benign Tumor or the cancer is not present"
        else:
            return "Cancer is present kindly consult a docter"

#######################################################################################################################
#######################################################################################################################

##Front end code

class APP(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Cancer Analyzer")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (IntroPage, UploadPage, RunPage, AboutPage, PredictPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(IntroPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class IntroPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Cancer Analyzer", font=('Veranda', 30))

        button = tk.Button(self, text="Prediction Test",
                            command=lambda: controller.show_frame(UploadPage))

        button2 = tk.Button(self, text="About",
                             command=lambda: controller.show_frame(AboutPage))
        

        global first_pass
        status = tk.Label(self, textvariable=first_pass)
        status.grid(row=3, columnspan=3, sticky=tk.E)
        label.grid(row=0, columnspan=3, pady=50, padx=450)
        
        button.grid(row=2, column=0, pady=20, padx=20)

        button2.grid(row=2, column=2, pady=20, padx=15)


class UploadPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="PID:", font=LARGE_FONT)
        self.entry = tk.Entry(self)

        def printcommand():
            global patient_id
            patient_id = int(self.entry.get())
            print(patient_id)

        button1 = tk.Button(self, text="Back",
                             command=lambda: controller.show_frame(IntroPage))


        button2 = tk.Button(self, text="OK",
                             command=printcommand)

        button3 = tk.Button(self, text="Run",
                             command=lambda: controller.show_frame(RunPage))
        global first_pass
        status = tk.Label(self, textvariable=first_pass)
        status.grid(row=3, columnspan=10, sticky=tk.E)
        label.grid(row=3, column=4, columnspan=2, pady=50, padx=450)
        self.entry.grid(row=3, column=3, columnspan=9, pady=50, padx=440)
        button1.grid(row=6, column=3, columnspan=2, pady=20, padx=20)
        button2.grid(row=6, column=6, columnspan=2, pady=20, padx=20)
        button3.grid(row=6, column=9, columnspan=2, pady=20, padx=20)

class RunPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="To Predict kindly press the predict button below", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Back",
                             command=lambda: controller.show_frame(IntroPage))
        button1.pack()

        button2 = tk.Button(self, text="Prediction",
                             command=lambda: controller.show_frame(PredictPage))
        button2.pack(side=tk.BOTTOM)
       

class AboutPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=abttxt, font=LARGE_FONT)
        label.pack(pady=150, padx=30)

        button1 = tk.Button(self, text="Home",
                             command=lambda: controller.show_frame(IntroPage))
        button1.pack(side=tk.BOTTOM)

    

class PredictPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Prediction", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        global prediction, patient_id
        button1 = tk.Button(self, text="Home",
                             command=lambda: controller.show_frame(IntroPage))

        global patient_id
        val = predict_result(patient_id)
        val = 'CNN predicted with 63% accuracy : ' + val
        label = tk.Label(self, text=val, font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        button1.pack()


app = APP()
app.geometry("1280x720")
app.mainloop()