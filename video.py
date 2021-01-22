from PIL import Image

from footprints.predict_simple import InferenceManager
import os
import cv2
import numpy as np
import torch
import argparse

from footprints.utils import sigmoid_to_depth


class VideoManager(InferenceManager):
	def predict_for_single_frame(self, image):
		"""Use the model to predict for a single image and save results to disk
		"""
		original_image, preprocessed_image = self._load_and_preprocess_image(image)
		pred = self.model_manager.model(preprocessed_image)
		pred = pred['1/1'].data.cpu().numpy().squeeze(0)

		hidden_ground = cv2.resize(pred[1], original_image.size) > 0.5
		hidden_depth = cv2.resize(sigmoid_to_depth(pred[3]), original_image.size)
		original_image = np.array(original_image) / 255.0

		# normalise the relevant parts of the depth map and apply colormap
		_max = hidden_depth[hidden_ground].max()
		_min = hidden_depth[hidden_ground].min()
		hidden_depth = (hidden_depth - _min) / (_max - _min)
		depth_colourmap = self.colormap(hidden_depth)[:, :, :3]  # ignore alpha channel

		# create and save visualisation image
		hidden_ground = hidden_ground[:, :, None]
		visualisation = original_image * (1 - hidden_ground) + depth_colourmap * hidden_ground
		return (visualisation[:, :, ::-1] * 255).astype(np.uint8)

	def predict_for_video(self, video_path):
		"""Use the model to predict for a single video and save results to disk
		"""
		cap = cv2.VideoCapture(video_path)

		# Define the codec and create VideoWriter object
		fourcc = cv2.VideoWriter_fourcc(*'X264')
		out = cv2.VideoWriter('output.mkv', fourcc, 15.0, (1280, 720))

		w = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
		h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
		print("cap: ", w, h)

		i = 0

		while cap.isOpened():
			ret, frame = cap.read()
			i += 1
			if i % 4 != 0:
				continue
			if ret:
				#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				#gray = cv2.merge((gray, gray, gray))
				#il write si aspetta una immagine 3-channel, la conversione in
				#grigio la trasforma in immagine 1-channel e il writer non riesce a scriverla. Potrei anche modificare
				#il writer in modo da supportare immagini 1-canale
				try:
					predict = self.predict_for_single_frame(frame)
					#print(predict.shape)
					if predict.shape == (720, 1280, 3):
						out.write(predict)
					else:
						print("Strana predict shape", predict.shape)
						out.write(frame)
				except ValueError:
					print("Frame", i, "error")
					out.write(frame)
				if i % 60 == 0:
					print("Secondo:", i/60)
					#print(frame.shape, gray.shape, predict.shape)
			else:
				print("Sono entrato nel break")
				break

		cap.release()
		out.release()

	def _load_and_preprocess_image(self, image):
		cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		original_image = Image.fromarray(cv2_image)
		preprocessed_image = self.resizer(original_image)
		preprocessed_image = self.totensor(preprocessed_image)
		preprocessed_image = preprocessed_image[None, ...]
		if self.use_cuda:
			preprocessed_image = preprocessed_image.cuda()
		return original_image, preprocessed_image

def parse_args():
	parser = argparse.ArgumentParser(
		description='Simple prediction from a footprints model.')

	parser.add_argument('--video', type=str,
						help='path to a test video or folder of videos', required=True)
	parser.add_argument('--model', type=str,
						help='name of a pretrained model to use',
						choices=["kitti", "matterport", "handheld"])
	parser.add_argument("--no_cuda",
						help='if set, disables CUDA',
						action='store_true')
	parser.add_argument("--save_dir", type=str,
						help='where to save npy and visualisations to',
						default="predictions")
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	videoManager = VideoManager(
		model_name=args.model,
		use_cuda=torch.cuda.is_available() and not args.no_cuda,
		save_visualisations=True,
		save_dir=args.save_dir)
	videoManager.predict_for_video(args.video)
