import os
import sys
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from util.util import tensor2im
from data.base_dataset import get_transform 

import PIL
from PIL import Image, ImageOps
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
import cv2
import numpy as np
import time
import torch

def read_classes(file_path):

	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()

	return classes

class Qt(QWidget):
	def mv_Chooser(self):    
		opt = QFileDialog.Options()
		opt |= QFileDialog.DontUseNativeDialog
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "C:/Users/hongze/Desktop/night/crop/","Mp4 (*.mp4)", options=opt)
	
		return fileUrl[0]

classes = read_classes('classes.txt')
num_frame = 10
imgSize = (224,224)

if __name__ == '__main__':

	#讀取影片路徑
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)

	if(not os.path.exists('./dataset')):
		for i in range(len(classes)):
			os.makedirs('./dataset/'+str(classes[i]))
		

	#開啟影片
	cap = cv2.VideoCapture(fileUrl)
	movie_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	

	#Define CUT
	cut_opt = TestOptions().parse()  # get test options
	cut_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
	cut_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
	cut_model = create_model(cut_opt)      # create a model given opt.model and other options
	cut_model.setup(cut_opt)               # regular setup: load and print networks; create schedulers
	cut_transform = get_transform(cut_opt)
	ts_b = torch.rand(5, 3)
	if cut_opt.eval:
		cut_model.eval()

	#圖片處理
	
	ret, frame = cap.read()
	i = 1
	series = []
	frame_fps5_num = 0
	while(ret):
		if(frame_fps5_num%6==0):
			#frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
			#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			#draw = gray.copy()
			#series.append(gray)
			draw = frame.copy()

			#調整訓練圖片大小
			frame = cv2.resize(frame, imgSize)
			gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

			data = {'A': cut_transform(Image.fromarray(gray_frame)).unsqueeze(0),'B':ts_b, 'A_paths': ['doesnt_really_matter'], 'B_paths': ['doesnt_really_matter']} 

			cut_model.set_input(data)  # unpack data from data loader
			cut_model.test()           # run inference
			visuals = cut_model.get_current_visuals()  # get image results

			im_data = list(visuals.items())[1][1] # grabbing the important part of the result
			cg_im = tensor2im(im_data)  # convert tensor to image

			cg_im = cv2.resize(cg_im, (224,224))

			series.append(cg_im)
			

			for c in range(len(classes)):
				cv2.putText(draw,str(c)+":"+classes[c],(20,20+c*40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255),1)
			cv2.putText(draw,str(len(classes))+":"+"X",(20,20+len(classes)*40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255),1)
			draw = cv2.resize(draw, (int(2*draw.shape[1]/3), int(2*draw.shape[0]/3)))
			cv2.imshow("src", draw)
			cv2.imshow("cg_im", cg_im)

			if(i%num_frame == 0):
				class_index = input()
				if(class_index == 'q'):
					exit(0)
				class_index = int(class_index)
				if(class_index<len(classes)):
					path = "./dataset/"+str(classes[class_index])
					file_num = len([lists for lists in os.listdir(path) if os.path.isdir(os.path.join(path, lists))])
					print("---",file_num+1)
					os.makedirs(path+"/"+str(classes[class_index])+"_"+str(file_num+1).zfill(6))
					for n in range(num_frame):
						#存圖片時檔名數字須從1開始(3Dresnet從編號1開始讀取)
						path = "./dataset/"+str(classes[class_index])+"/"+str(classes[class_index])+"_"+str(file_num+1).zfill(6)+"/"+"image_"+str(n+1).zfill(5)+".jpg"
						cv2.imwrite(path, series[n])

				series = []
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break
			i = i + 1
		ret, frame = cap.read()
		frame_fps5_num = frame_fps5_num+1
	cap.release()
	cv2.destroyAllWindows()

		
		