import argparse

from footprints.predict_simple import InferenceManager, parse_args
import torch
import os
import cv2
from footprints.utils import sigmoid_to_depth
import numpy as np
from scipy import ndimage
from matplotlib import colors as clr

import posenet

from sklearn.cluster import DBSCAN


class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	@staticmethod
	def createFromList(coords):
		return Point(coords[0], coords[1])

	def getAbsoluteDistance(self, otherPoint):
		return pow((self.getXFloat() - otherPoint.getXFloat()) ** 2 + (self.getYFloat() - otherPoint.getYFloat()) ** 2,  1 / 2)

	def getClosestPoint(self, otherPoints):
		closestDist = 10000
		for i in range(len(otherPoints)):
			dist = self.getAbsoluteDistance(otherPoints[i])
			if dist < closestDist:
				closest = otherPoints[i]
				closestDist = dist
		return closest

	def getMidPoint(self, otherPoint):
		return Point(abs((self.getXFloat() + otherPoint.getXFloat()) / 2), abs((self.getYFloat() + otherPoint.getYFloat()) / 2))

	def getXInt(self):
		return int(round(self.x))

	def getYInt(self):
		return int(round(self.y))

	def getXFloat(self):
		return float(self.x)

	def getYFloat(self):
		return float(self.y)

	def __str__(self):
		return "[" + str(self.getXFloat()) + ", " + str(self.getYFloat()) + "]"

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


def find_feet_clusters_dbscan(feet_coords, color=None, visualisation=None):
	labels = list(DBSCAN(eps=35, min_samples=2).fit(feet_coords).labels_)

	feet_clusters = {el: [] for el in set(labels)}

	for i in range(len(labels)):
		feet_clusters[labels[i]].append(feet_coords[i])

	if color is not None and visualisation is not None:
		print("Feet clusters:", feet_clusters)

	people_coords = list(feet_clusters[-1])
	for label in feet_clusters:
		if label >= 0:
			if len(feet_clusters[label]) > 2:
				print("In questo cluster ci sono", len(feet_clusters[label]), "punti:", feet_clusters[label])

			p = Point.createFromList(feet_clusters[label][0]).getMidPoint(Point.createFromList(feet_clusters[label][1]))
			people_coords.append([p.getXInt(), p.getYInt()])

			if color is not None and visualisation is not None:
				visualisation = draw_points(visualisation, [p], radius=3, colorPoints=color)

	return feet_clusters, people_coords, visualisation


def find_people_clusters_dbscan(people_coords, colors=None, visualisation=None):
	labels = list(DBSCAN(eps=80, min_samples=2).fit(people_coords).labels_)

	people_clusters = {el: [] for el in set(labels)}

	for i in range(len(labels)):
		people_clusters[labels[i]].append(people_coords[i])

	if colors is not None and visualisation is not None:
		print("People clusters:", people_clusters)
		i = 0
		for label in people_clusters:
			if label >= 0:
				color = clr.to_rgba(colors[i % len(colors)])
				visualisation = draw_points(visualisation, people_clusters[label], radius=3, colorPoints=color)
				i += 1

	return people_clusters, visualisation


# funzioni per disegnare
def draw_points(img, points, radius=2, colorPoints=clr.to_rgba('white')):
	for point in points:
		if isinstance(point, Point):
			x = point.getXInt()
			y = point.getYInt()
		elif isinstance(point, list):
			x, y = point
		img[x - radius:x + radius, y - radius:y + radius, 0:2] = colorPoints[0:2]
	return img


def draw_line(img, pointA, pointB, colorLine=clr.to_rgba('blue')):
	return cv2.line(img, (pointA.getYInt(), pointA.getXInt()), (pointB.getYInt(), pointB.getXInt()), color=colorLine,
					thickness=1)


def draw_tag(img, point, text, colorText=clr.to_rgba('red')):
	return cv2.putText(img=img, text=text, org=(point.getYInt(), point.getXInt()), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					   color=colorText, thickness=1, fontScale=0.6)


def draw_line_with_tag(img, pointA, pointB, text, colorLine=clr.to_rgba('blue'), colorText=clr.to_rgba('red')):
	img = cv2.line(img, (pointA.getYInt(), pointA.getXInt()), (pointB.getYInt(), pointB.getXInt()), color=colorLine,
				   thickness=1)
	midp = pointA.getMidPoint(pointB)
	img = cv2.putText(img=img, text=text, org=(midp.getYInt(), midp.getXInt()), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					  color=colorText, thickness=1, fontScale=0.6)
	return img


def draw_distance(img, points, maxDistance, tag=True):
	dim = len(points)
	for i in range(dim):
		for j in range(i + 1, dim):
			# dist = getAbsoluteDistance(points[i], points[j]) perche' sono scambiate la x e la y? non esiste motivo logico
			dist = points[i].getAbsoluteDistance(points[j])
			if dist <= float(maxDistance):
				if tag:
					img = draw_line_with_tag(img, points[i], points[j], str(int(dist)))
				else:
					img = draw_line(img, points[i], points[j])
	return img


def draw_info_about_the_closest(img, points, maxDistance):
	dim = len(points)
	for i in range(dim):
		closest = points[i].getClosestPoint(points[:i] + points[i + 1:])
		dist = points[i].getAbsoluteDistance(closest)
		img = draw_tag(img, points[i], str(int(dist)))
	return img


# funzioni di calcolo
def onePointEachPerson(centers_of_mass, maxDistance):
	result = []
	alreadyDone = []
	dim = len(centers_of_mass)
	for i in range(dim):
		currentMax = maxDistance
		if i not in alreadyDone:
			point = centers_of_mass[i]
			done = None
			for j in range(i + 1, dim):
				dist = centers_of_mass[i].getAbsoluteDistance(centers_of_mass[j])
				if dist < currentMax:
					currentMax = dist
					point = centers_of_mass[i].getMidPoint(centers_of_mass[j])
					done = j
			result.append(point)
			if done:
				alreadyDone.append(done)
	return result


class ObstacleManager(InferenceManager):
	def __init__(self, model_name, save_dir, use_cuda, save_visualisations=True):
		super().__init__(model_name, save_dir, use_cuda, save_visualisations)
		self.posenet_model = posenet.load_model(args.posenet_model)
		if self.use_cuda:
			self.posenet_model = self.posenet_model.cuda()
		else:
			self.posenet_model = self.posenet_model.cpu()
		self.output_stride = self.posenet_model.output_stride

	def posenet_predict(self, filename, base_out_image=None):
		input_image, draw_image, output_scale = posenet.read_imgfile(
			filename, scale_factor=args.scale_factor, output_stride=self.output_stride)

		with torch.no_grad():
			if self.use_cuda:
				input_image = torch.Tensor(input_image).cuda()
			else:
				input_image = torch.Tensor(input_image).cpu()

			heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.posenet_model(input_image)

			pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
				heatmaps_result.squeeze(0),
				offsets_result.squeeze(0),
				displacement_fwd_result.squeeze(0),
				displacement_bwd_result.squeeze(0),
				output_stride=self.output_stride,
				max_pose_detections=10,
				min_pose_score=0.25)

		keypoint_coords *= output_scale

		if self.save_visualisations:
			base_out_image = draw_image if base_out_image is None else base_out_image

			draw_image = posenet.draw_skel_and_kp(
				base_out_image, pose_scores, keypoint_scores, keypoint_coords,
				min_pose_score=0.25, min_part_score=0.25)

			if base_out_image is None:
				vis_save_path = os.path.join(self.save_dir, "visualisations", "posenet_ " + os.path.basename(filename) + '.jpg')
				cv2.imwrite(vis_save_path, draw_image)
				print("Image saved to", vis_save_path)

		if not args.notxt:
			print()
			print("Results for image: %s" % filename)
			for pi in range(len(pose_scores)):
				if pose_scores[pi] == 0.:
					break
				print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
				for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
					print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

		return draw_image


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

			# trovo i baricentri dei clusters e li associo all'immagine
			points = [Point(clusterInfo.com[0], clusterInfo.com[1]) for clusterInfo in clustersInfo.values() if
							clusterInfo.isFoot]
			visualisation = draw_points(visualisation, points, radius=1)

			feet_coords = [[point.getXInt(), point.getYInt()] for point in points]

			# a partire dai baricentri accoppio i piedi identificando le persone e associo questi punti all'immagine
			peoplePoints = onePointEachPerson(points, 31)  # massima distanza tollerabile tra i piedi
			visualisation = draw_points(visualisation, peoplePoints, colorPoints=clr.to_rgba('yellow'))

			colors = ["orange", "green", "blue", "chocolate", "dimgrey", "black"]

			feet_clusters, people_coords_dbscan, visualisation = find_feet_clusters_dbscan(feet_coords, clr.to_rgba('red'), visualisation)

			# associo all'immagine le linee che uniscono le persone con tag riferito a distanza
			visualisation = draw_distance(img=visualisation, points=peoplePoints, maxDistance=100)

			people_clusters_dbscan, visualisation = find_people_clusters_dbscan(people_coords_dbscan, colors, visualisation)

			# associo all'immagine un tag per ogni persona con scritto la distanza della persona piu vicina
			# visualisation = draw_info_about_the_closest(img=visualisation, points=peoplePoints, maxDistance=100)

			visualisation = (visualisation[:, :, ::-1] * 255).astype(np.uint8)

			visualisation = self.posenet_predict(image_path, visualisation)

			print("└> Saving visualisation to {}".format(vis_save_path))
			cv2.imwrite(vis_save_path, visualisation)


def posenet_params(parser: argparse.ArgumentParser):
	parser.add_argument("--posenet_model", type=int, default=101)
	parser.add_argument('--scale_factor', type=float, default=1.0)
	parser.add_argument('--notxt', action='store_true')


if __name__ == '__main__':
	args = parse_args(posenet_params)
	inference_manager = ObstacleManager(
		model_name=args.model,
		use_cuda=torch.cuda.is_available() and not args.no_cuda,
		save_visualisations=not args.no_save_vis,
		save_dir=args.save_dir)
	inference_manager.predict(image_path=args.image)



