'''
由自己的手写数据生成 images 和 labels 并保存
'''


import os
from PIL import Image
import numpy as np


class NoDataError(Exception):
	pass


def img_box(img):
	index=np.where(img)
	weight=img[index]
	index_x_mean=np.average(index[1],weights=weight)
	index_y_mean=np.average(index[0],weights=weight)
	index_xy_std=np.sqrt(np.average(((index[1]-index_x_mean)**2+
					(index[0]-index_y_mean)**2)/2,weights=weight))
	box=(index_x_mean-3*index_xy_std,index_y_mean-3*index_xy_std,
			index_x_mean+3*index_xy_std,index_y_mean+3*index_xy_std)	
	return box
	
	
def normalize_image(image):
	img0=255-np.array(image)
	img1=np.where(img0>=100,img0,0)
	img2=Image.fromarray(img1)
	box=img_box(img1)
	crop_img=img2.crop(box)
	norm_img=crop_img.resize((28,28))
	norm_array=np.array(norm_img).flatten()/255
	return norm_array
	
	
def format_handwriting(dirpath):
	origin_images,images,labels=[],[],[]
	for num in range(10):
		dirname='write_{}'.format(num)
		filepath=os.path.join(dirpath,dirname)
		for filename in os.scandir(filepath):
			if filename.is_file():
				origin_img=Image.open(os.path.join(filename))
				image_fig=Image.open(os.path.join(filename)).convert('L')
				image_ary=normalize_image(image_fig)
				origin_images.append(np.array(origin_img))
				images.append(image_ary)
				labels.append(num)
	origin_images=np.array(origin_images)
	images=np.array(images)
	labels=np.array(labels)
	return origin_images,images,labels
	
	
def save_handwriting(dirpath,filename='handwriting.npz'):
	origin_images,images,labels=format_handwriting(dirpath)
	filepath=os.path.join(dirpath,'npz')
	file=os.path.join(filepath,filename)
	np.savez(file,origin_images=origin_images,images=images,labels=labels)
	
	
def load_handwriting(dirpath,filename='handwriting.npz'):
	filepath=os.path.join(dirpath,'npz')
	file=os.path.join(filepath,filename)
	if os.path.exists(file):
		handwriting=np.load(file, allow_pickle=True)
		return handwriting
	else:
		raise NoDataError(f'手写数据文件：{os.path.join(filepath,filename)} 不存在')

		
def main():
	save_handwriting('../../data/my_handwriting')


if __name__=='__main__':
	main()
