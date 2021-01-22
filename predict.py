from footprints.predict_simple import InferenceManager, parse_args
import torch
import os
import cv2
from footprints.utils import sigmoid_to_depth
import numpy as np
from scipy import ndimage


class ClusterInfo:
	def __init__(self, valore, dimensione, isFoot):
		self.valore = valore
		self.dimensione = dimensione
		self.isFoot = isFoot
		self.com = None


def find_clusters(array):
	clustered = np.empty_like(array, dtype=np.int)
	feet = np.zeros_like(array, dtype=np.bool)
	unique_vals = np.unique(array)
	cluster_count = 0
	clustersInfo = {}
	ones = np.ones_like(array, dtype=int)
	for val in unique_vals:
		labelling, label_count = ndimage.label(array == val)
		for k in range(1, label_count + 1):
			clustered[labelling == k] = cluster_count
			dimensione = len(clustered[labelling == k])
			isFoot = False
			if val == False and dimensione < 1000:  # val è del tipo np.bool_ quindi non posso usare val is False
				feet[labelling == k] = True
				isFoot = True
			clustersInfo[cluster_count] = ClusterInfo(val, dimensione, isFoot)
			cluster_count += 1

	coms = ndimage.center_of_mass(ones, labels=clustered, index=range(cluster_count))
	for i in range(cluster_count):
		clustersInfo[i].com = coms[i]
		print("Cluster #{}: {} elementi '{}' at {}".format(i, clustersInfo[i].dimensione, clustersInfo[i].valore, clustersInfo[i].com))
	return clustered, cluster_count, clustersInfo, feet


def findNearest(center_of_mass):
	return int(round(center_of_mass[0])), int(round(center_of_mass[1]))


def getAbsoluteDistance(pointA, pointB):  # pointA[0] e' la x, pointB[1] e' la y
	return pow((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2, 1 / 2)


def midpoint(pointA, pointB):
	return [abs((pointA[0] + pointB[0]) / 2), abs((pointA[1] + pointB[1]) / 2)]


def onePointEachPerson(clustersInfo, maxDistance):
	result = []
	alreadyDone = []
	dim = len(clustersInfo)
	for i in range(dim):
		max = maxDistance
		if i not in alreadyDone and clustersInfo[i].isFoot:
			point = clustersInfo[i].com
			done = None
			for j in range(i + 1, dim):
				dist = getAbsoluteDistance(clustersInfo[i].com, clustersInfo[j].com)
				if dist < max:
					max = dist
					point = midpoint(clustersInfo[i].com, clustersInfo[j].com)
					done = j
			result.append(point)
			if done:
				alreadyDone.append(done)
	return result


class ObstacleManager(InferenceManager):
	def predict_for_single_image(self, image_path):
		"""Use the model to predict for a single image and save results to disk
		"""
		print("Predicting for {}".format(image_path))
		original_image, preprocessed_image = self._load_and_preprocess_image(image_path)
		pred = self.model_manager.model(preprocessed_image)
		pred = pred['1/1'].data.cpu().numpy().squeeze(0)

		filename, _ = os.path.splitext(os.path.basename(image_path))
		npy_save_path = os.path.join(self.save_dir, "outputs", filename + '.npy')
		print("└> Saving predictions to {}".format(npy_save_path))
		np.save(npy_save_path, pred)

		if self.save_visualisations:
			# print(pred[1].shape, pred.shape)
			# tutti i pred[0 -> 3] hanno shape (256, 448)
			hidden_ground = cv2.resize(pred[1], original_image.size) > 0.95
			print(hidden_ground.shape)
			clusters, numeroCluster, clustersInfo, feet = find_clusters(hidden_ground)
			feet = np.expand_dims(feet, axis=2).astype(np.int)
			print(feet.shape)
			hidden_depth = cv2.resize(sigmoid_to_depth(pred[3]), original_image.size)
			# TODO: da qui indentificare il centro di ogni footprint e vedere la distanza nello stesso punto della
			# hidden_depth in modo da vedere quanto è distante nella scena
			original_image = np.array(original_image) / 255.0

			# normalise the relevant parts of the depth map and apply colormap
			_max = hidden_depth[hidden_ground].max()
			_min = hidden_depth[hidden_ground].min()
			hidden_depth = (hidden_depth - _min) / (_max - _min)
			depth_colourmap = self.colormap(hidden_depth)[:, :, :3]  # ignore alpha channel

			# create and save visualisation image
			hidden_ground = hidden_ground[:, :, None]
			# visualisation = original_image * (1 - hidden_ground)# + depth_colourmap * hidden_ground
			visualisation = original_image * (1 - hidden_ground) + depth_colourmap * hidden_ground
			# visualisation = original_image * 0.05 + depth_colourmap * 0.95
			# on = np.ones(shape=(682, 1024, 1))
			# off = np.zeros(shape=(682, 1024, 1))
			# colors = np.concatenate((on, off, off), axis=2)
			# visualisation = original_image * (1 - feet) + feet * colors #np.ones(shape=original_image.shape)
			visualisation = original_image * (1 - feet) + feet * depth_colourmap
			vis_save_path = os.path.join(self.save_dir, "visualisations", filename + '.jpg')
			print(visualisation.shape)

			for i in clustersInfo:
				x, y = findNearest(clustersInfo[i].com)
				visualisation[x - 1:x + 1, y - 1:y + 1, 0] = 0
				visualisation[x - 1:x + 1, y - 1:y + 1, 1:2] = 1

			print("POINTS:")
			points = onePointEachPerson(clustersInfo, 31)  # massima distanza tollerabile tra i piedi
			for p in points:
				x, y = findNearest(p)
				print("(" + str(x) + "," + str(y) + ")")
				visualisation[x - 2:x + 2, y - 2:y + 2, 0:2] = 1
				cv2.imwrite(vis_save_path, (visualisation[:, :, ::-1] * 255).astype(np.uint8))

			print("└> Saving visualisation to {}".format(vis_save_path))
			cv2.imwrite(vis_save_path, (visualisation[:, :, ::-1] * 255).astype(np.uint8))


if __name__ == '__main__':
	args = parse_args()
	inference_manager = ObstacleManager(
		model_name=args.model,
		use_cuda=torch.cuda.is_available() and not args.no_cuda,
		save_visualisations=not args.no_save_vis,
		save_dir=args.save_dir)
	inference_manager.predict(image_path=args.image)



