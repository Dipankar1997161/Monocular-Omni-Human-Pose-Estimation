import sys
# Set paths for CenterNet files
CENTERNET_LIB_PATH = "/home/ndip/CenterNet/src/lib/"
CENTERNET_SRC_PATH = "/home/ndip/CenterNet/src/"
sys.path.insert(0, CENTERNET_LIB_PATH)
sys.path.insert(1, CENTERNET_SRC_PATH)

# Import libraries
import os
import cv2
import numpy as np
import pandas as pd
from opts import opts
from detectors.detector_factory import detector_factory
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = ' 1'

# Supported image extensions
image_ext = ['jpg', 'jpeg', 'png', 'webp']
# Supported video extensions
video_ext = ['mp4', 'mov', 'avi', 'mkv']
# Name of time stats prints
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

# Colors for located keypoints
colors_hp = [(255, 0, 255), (161, 169, 210),(161, 169, 210), 
			(161, 169, 210), (161, 169, 210), (255, 0, 0), (0, 0, 255),
			(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
			(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
			(255, 0, 0), (0, 0, 255)]

# Colors for skeleton
ec = [(255, 0, 0), (0, 0, 255), (255, 0, 255),
		(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
		(255, 0, 0), (0, 0, 255), (255, 0, 255),
		(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]

# Links between keypoints, for skeleton construction
edges = [[0,5], [0,6], [5, 6], 
		[5, 7], [7, 9], [6, 8], [8, 10], 
		[5, 11], [6, 12], [11, 12], 
		[11, 13], [13, 15], [12, 14], [14, 16]]

def add_coco_bbox(img,bbox, conf=1, show_txt=True):
	"""
	draws bounding box over img
	-----
	Params
	-----
	img: np.array
		input image
	bbox: list
		bounding box coordinates
	conf: float
		confidence in detection
	show_txt: bool
		show text with confidence score and category over bbox
	------
	Returns
		Image with bounding box drawn over
	"""
	bbox = np.array(bbox, dtype=np.int32)
	c = [0,255,0] # bbox color
	txt = '{}{:.1f}'.format("person", conf) # text to display
	font = cv2.FONT_HERSHEY_SIMPLEX
	cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
	# Creates rectangle over image
	cv2.rectangle(
		img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
	# Draws text over image, if requested
	if show_txt:
	  cv2.rectangle(img,
	                (bbox[0], bbox[1] - cat_size[1] - 2),
	                (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
	  cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
	              font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
	return img

def add_coco_hp(img,points): 
	"""
	draws detected keypoints and skeleton over input image
	-----
	Params
	-----
	img: np.array
		input image
	points: list
		keypoint coordinates
	------
	Returns
		Image with keypoints and skeletons drawn over
	"""
	points = np.array(points, dtype=np.int32).reshape(17, 2)
	# Draws each keypoint over image
	for j in range(17):
		cv2.circle(img,
				(points[j, 0], points[j, 1]), 3, colors_hp[j], -1)
	# Draws lines joining keypoints, to form skeletons
	for j, e in enumerate(edges):
		if points[e].min() > 0:
			cv2.line(img, (points[e[0], 0], points[e[0], 1]),
			(points[e[1], 0], points[e[1], 1]), ec[j], 2,
  	lineType=cv2.LINE_AA)
	return img

def draw_detection(img,results,min_confidence):
	"""
	draws detected bounding box and keypoints over image
	-----
	Params
	-----
	img: np.array
		input image
	results: dict
		dictionary with keypoint and bounding box detections
	min_confidence: float
		minimum confidence in detection for displaying
	------
	Returns
		Image with keypoints and bounding boxes drawn over
	"""
	for bbox in results[1]:
		# Verifies if detection is over threshold
		if bbox[4] > min_confidence:
			ret_img = add_coco_bbox(img,bbox[:4], bbox[4]) # draws bbox
			ret_img = add_coco_hp(ret_img,bbox[5:39])         # draws kpts
    # If detection is not over confidence threshold, returns original image
		else:
			ret_img = img
	return ret_img

def org_detections(results,min_confidence):
	"""
	returns detection results as an structured dataframe
	-----
	Params
	-----
	results: dict
		dictionary with keypoint and bounding box detections
	min_confidence: float
		minimum confidence in detection for displaying
	------
	Returns
		Dataframe with bounding box coordinates, score and keypoint locations
	"""
	columns = ["topleft_bbox","botright_bbox","score","nose","left_shoulder",
			   "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee",
			   "right_knee","left_ankle","right_ankle"]
	df = pd.DataFrame(columns=columns)
	det_idx = 0
	for bbox in results[1]:
		# Only saves detections over threshold confidence
		if bbox[4] > min_confidence:
			# Boundix box coordinates
			topleft_bbox = [(bbox[0],bbox[1])]
			botright_bbox = [(bbox[2],bbox[3])]
			# Detection score
			score = [bbox[4]]
			# Keypoints coordinates
			x_kpts = bbox[5:32:2]
			y_kpts = bbox[6:32:2]
			xy_kpts = list(zip(x_kpts,y_kpts))
			det = topleft_bbox + botright_bbox + score + xy_kpts
			# Appends detection info to dataframe
			df.loc[det_idx] = det
			det_idx += 1
	return df




def main(args):
	# Selects appropiate paths according to backbone selection
	if args.arch == 'dla':
		MODEL_PATH = "/home/ndip/CenterNet/exp/multi_pose/dla_transfer_merged/model_last.pth"
		arch_name = 'dla_34'
	elif args.arch == 'hourglass':
		MODEL_PATH = "/home/ndip/CenterNet/exp/multi_pose/hg_transfer_hourglass/model_last.pth"
		arch_name = args.arch
	elif args.arch =='hrnet':
		MODEL_PATH = "../CenterNet/models/multi_pose_hrnet_3x_gray_finetune"
		arch_name = 'hrnet32'
	# Initializes centernet options
	opt = opts().init('{} --load_model {} --arch {}'.format('multi_pose', MODEL_PATH,arch_name).split(' '))
	# Creates detector
	detector = detector_factory[opt.task](opt)
	detector.pause = False


	if args.demo == 'webcam' or \
		args.demo[args.demo.rfind('.') + 1:].lower() in video_ext:
		# Initializes video capture for frame retrieval
		cam = cv2.VideoCapture(0 if args.demo == 'webcam' else args.demo)
		# In case output directory is specified, creates video writer for saving each frame into output video
		if args.output_dir != '' and args.save_img:
			output_video_path = '{}{}_{}.mp4'.format(args.output_dir,args.demo.split("/")[-1].split(".")[0],args.arch)
			_, sample_img = cam.read()
			out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('M','J','P','G'), 16, 
				  (sample_img.shape[0],sample_img.shape[1]))
		frame_idx = 0
		while True:
			_, img = cam.read()        # reads frame from webcam or video
			frame_idx += 1
			ret = detector.run(img)    # runs detection over input image
			ret_img = draw_detection(img.copy(),ret["results"],args.min_confidence) # draws detection over input image
			# If user wants, fps can be shown over detection image
			if args.show_fps:
				cv2.putText(ret_img,'fps: {:.2f}'.format((1/ret['tot'])),(0,30), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 255), 2, cv2.LINE_AA)
			if args.visualize:
				cv2.imshow('entrada', img) # shows input image
				cv2.imshow('deteccion',ret_img) # shows image with detections
			# Writes frame with detection over output video
			if args.output_dir != '':
				if args.save_img:
					out.write(ret_img)
				if args.save_csv:
					df_det = org_detections(ret["results"],args.min_confidence)
					df_det.to_csv("{}{}-{}_{}.csv".format(args.output_dir,args.demo.split("/")[-1].split(".")[0],frame_idx,args.arch))
			# Prints time stats
			time_str = ''
			for stat in time_stats:
				time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
			print(time_str)
			# Option for exiting program
			if cv2.waitKey(0 if args.pause else 1) == 27:
				cam.release()
				out.release()
				import sys
				sys.exit(0)
		cam.release()
		out.release()

	else:
		# If demo is image or directory with images, retrieves path for each one of them
		if os.path.isdir(args.demo):
			image_names = []
			ls = os.listdir(args.demo)
			print(args.demo)
			for file_name in sorted(ls):
				ext = file_name[file_name.rfind('.') + 1:].lower()
				if ext in image_ext:
			 		image_names.append(os.path.join(args.demo, file_name))
		else:
			image_names = [args.demo]

		for (image_name) in image_names:
			# Reads image
			img = cv2.imread(image_name)
			ret = detector.run(img)    # runs detection over image
			ret_img = draw_detection(img.copy(),ret["results"],args.min_confidence) # draws detections over image
			if args.visualize:
				cv2.imshow('entrada', img) # shows input image
				cv2.imshow('deteccion',ret_img) # shows image with detections

			# saves output image with detections, if requested
			if args.output_dir != '':
				if args.save_img:
					output_img_path = '{}{}_{}.png'.format(args.output_dir,image_name.split("/")[-1].split(".")[0],args.arch)
					cv2.imwrite(output_img_path,ret_img)
				if args.save_csv:
					df_det = org_detections(ret["results"],args.min_confidence)
					df_det.to_csv("{}{}_{}.csv".format(args.output_dir,image_name.split("/")[-1].split(".")[0],args.arch))
			# Option for exiting program
			if cv2.waitKey(0 if args.pause else 1) == 27:
				import sys
				sys.exit(0)
			# Prints time stats
			time_str = ''
			for stat in time_stats:
				time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
			print(time_str)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--demo', default='', 
                         help='path to image/ image folders/ video. ')

	parser.add_argument('--pause', action='store_true', 
                         help='whether to pause between detections')

	parser.add_argument('--arch', default='dla', 
                             help='model architecture. Currently tested'
                                  'dla | hourglass | hrnet')

	parser.add_argument('--min_confidence', type=float, default=0.3,
                             help='minimum confidence for visualization')

	parser.add_argument('--show_fps', action='store_true',
                             help='show fps of detection in visualization')

	parser.add_argument('--output_dir',type=str,default='',
							 help='output directory for detections')

	parser.add_argument('--save_img',action='store_true',
							 help='store images with detections')

	parser.add_argument('--save_csv',action='store_true',
							 help='save csv files with detected joints and bboxes')

	parser.add_argument('--visualize',type=int, default=1,
							 help='wheter to visualize outputs')

	args = parser.parse_args()
	main(args)
