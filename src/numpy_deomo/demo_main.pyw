#实现一个手写画板


import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel
from PyQt5.QtGui import QPainter, QPen, QFont, QPixmap
from PyQt5.QtCore import Qt 
from mnist_train import load_model
from py_model.NN_model import *


def img_box(img):
	index=np.where(img)
	weight=img[index]
	index_x_mean=np.average(index[1],weights=weight)
	index_y_mean=np.average(index[0],weights=weight)
	index_xy_std=np.sqrt(np.average(((index[1]-index_x_mean)**2+(index[0]-index_y_mean)**2)/2,
							weights=weight))
	box=(index_x_mean-3*index_xy_std,index_y_mean-3*index_xy_std,index_x_mean+3*index_xy_std,
			index_y_mean+3*index_xy_std)	
	return box
	
def normalize_figure():
	image0=Image.open('../data/GUI_fig/handwriting.png').convert('L')
	img=np.array(image0)
	img=np.where(img>=100,img,0)
	image=Image.fromarray(img)
	box=img_box(img)
	crop_img=image.crop(box)
	norm_img=crop_img.resize((28,28))
	return np.array(norm_img).reshape(1,-1)/255
	'''
	plt.imshow(norm_img,cmap='Greys',interpolation='nearest')
	plt.show()
	'''
	
class Palette(QMainWindow):
	def __init__(self):
		super().__init__()
		self.initUI()
		
	def initUI(self):
		self.setMouseTracking(False)
		self.pos_xy=[(-1,-1)]
		
		self.statusbar=self.statusBar()
		self.statusbar.setFont(QFont('Roman times',14))
		self.statusbar.showMessage('准备')
		
		act1=QAction('重写',self)
		act1.setStatusTip('清空全部笔画')
		act1.triggered.connect(self.clear)
		act2=QAction('撤回',self)
		act2.triggered.connect(self.back)
		act2.setStatusTip('撤销当前笔画')
		act3=QAction('识别',self)
		act3.setStatusTip('识别当前数字')
		act3.triggered.connect(self.identify)
		
		menubar=self.menuBar()
		menubar.addAction(act1)
		menubar.addAction(act2)
		menubar.addAction(act3)
		menubar.setFont(QFont('Roman times',16))
		
		self.lb=QLabel(self)
		self.lb.setGeometry(30,50,300,500)
		self.lb.setAlignment(Qt.AlignCenter)
		self.lb.setFont(QFont('Roman times',150))
		self.lb.setText('?')
		
		self.print_=False
		
		self.setGeometry(300,300,800,600)
		self.setWindowTitle('手写画板')
		self.setStyleSheet('QLabel{background-color:rgb(220,220,200)}')
		self.show()
		
	def paintEvent(self,event):
		qp1=QPainter()
		qp1.begin(self)
		self.handWrite1(qp1)
		qp1.end()
		if self.print_ is True:
			pix=QPixmap()
			pix.load('../data/GUI_fig/background.png')
			qp2=QPainter()
			qp2.begin(pix)
			self.handWrite2(qp2)
			qp2.end()
			pix.save('../data/GUI_fig/handwriting.png')
			self.print_=False
	
	def handWrite1(self,qp):
		qp.drawRect(29,49,301,501)
		qp.drawRect(370,50,400,500)
		
		pen=QPen(Qt.black,30,Qt.SolidLine)
		pen.setCapStyle(Qt.RoundCap)
		qp.setPen(pen)
		
		if len(self.pos_xy)>1:
			point_start=self.pos_xy[0]
			for pos_tem in self.pos_xy:
				point_end=pos_tem
				if point_start==(-1,-1) or point_end==(-1,-1):
					point_start=point_end
					continue 
				qp.drawLine(point_start[0],point_start[1],point_end[0],point_end[1])
				point_start=point_end
				
	def handWrite2(self,qp):
		pen=QPen(Qt.white,30,Qt.SolidLine)
		pen.setCapStyle(Qt.RoundCap)
		qp.setPen(pen)
		
		if len(self.pos_xy)>1:
			point_start=self.pos_xy[0]
			for pos_tem in self.pos_xy:
				point_end=pos_tem
				if point_start==(-1,-1) or point_end==(-1,-1):
					point_start=point_end
					continue 
				qp.drawLine(point_start[0]-370,point_start[1]-50,
							point_end[0]-370,point_end[1]-50)
				point_start=point_end
		
	def mouseMoveEvent(self,event):
		x,y=event.x(),event.y()
		if 370<x<370+400 and 50<y<50+500:
			pos_tem=(x,y)
			self.pos_xy.append(pos_tem)
			self.statusbar.showMessage('正在输入 ...')
			self.lb.setText('?')
		else:
			pos_stop=(-1,-1)
			if self.pos_xy[-1]!=pos_stop:
				self.pos_xy.append(pos_stop)
			self.statusbar.showMessage('准备')
		self.update()
		
	def mouseReleaseEvent(self,event):
		pos_stop=(-1,-1)
		if self.pos_xy[-1]!=pos_stop:
			self.pos_xy.append(pos_stop)
		self.statusbar.showMessage('准备')
		self.update()
		
	def clear(self):
		self.pos_xy=[(-1,-1)]		
		self.update()
		
	def back(self):
		if self.pos_xy==[(-1,-1)]:
			return
		pos_xy=self.pos_xy[::-1]
		index_xy=len(pos_xy)-pos_xy.index((-1,-1),1)
		self.pos_xy=self.pos_xy[:index_xy]
		self.update()
		
	def identify(self):
		self.print_=True 
		self.repaint()
		if len(self.pos_xy)>1:
			img=normalize_figure()
			img_pred=nn.predict(img)[0]
			self.lb.setText(str(img_pred))
		else:
			self.lb.setText('?')
		
		
if __name__=='__main__':
	nn=load_model()
	app=QApplication(sys.argv)
	palette=Palette()
	sys.exit(app.exec_())
	
					