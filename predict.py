import argparse

from footprints.predict_simple import InferenceManager, parse_args
import torch
import os
import cv2
from footprints.utils import sigmoid_to_depth
import numpy as np
from scipy import ndimage
from matplotlib import colors as clr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime

import posenet

from sklearn.cluster import DBSCAN


class Time:
	def __init__(self, save_dir, img_name, use_cuda, opt_level, img_shape=None):
		self.save_dir = save_dir
		self.start = time.time()
		self.img_name = img_name
		self.use_cuda = use_cuda
		self.opt_level = opt_level
		self.img_shape = img_shape
		self.steps = []

	def add_step(self, text):
		self.steps.append([time.time(), text])

	def write_info(self):
		shape = " " + str(self.img_shape[0]) + "x" + str(self.img_shape[1]) if self.img_shape is not None else ""
		device = "GPU" if self.use_cuda else "CPU"
		output_string = self.img_name + " [" + str(self.opt_level) + shape + " on " + device + "] ("\
			+ str(datetime.fromtimestamp(self.start)) + "):\n"
		if self.steps:
			output_string += "-> " + self.steps[0][1] + ": " + str(self.steps[0][0] - self.start) + "\n"
			for i in range(1, len(self.steps)):
				output_string += "-> " + self.steps[i][1] + ": " + str(self.steps[i][0] - self.steps[i-1][0]) + "\n"
			output_string += "TOTAL TIME ELAPSED: %f\n" % (self.steps[len(self.steps) - 1][0] - self.start)
		output_string += "\n"

		vis_save_path = os.path.join(self.save_dir, "execution_time.txt")
		open(vis_save_path, "a").write(output_string)
		print(output_string)


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


class Draw:
	def __init__(self, img):
		self.img = img
		self.stars_centers = []

	def get_img(self):
		return self.img

	def points(self, points, radius=2, colorPoints=clr.to_rgba('white')):
		for point in points:
			if isinstance(point, Point):
				x = point.getXInt()
				y = point.getYInt()
			elif isinstance(point, list):
				x, y = point
			self.img[x - radius:x + radius, y - radius:y + radius, 0:2] = colorPoints[0:2]

	def line(self, pointA, pointB, colorLine=clr.to_rgba('blue')):
		self.img = cv2.line(self.img, (pointA.getYInt(), pointA.getXInt()), (pointB.getYInt(), pointB.getXInt()), color=colorLine,
						thickness=1)

	def tag(self, point, text, colorText=clr.to_rgba('red')):
		self.img = cv2.putText(img=self.img, text=text, org=(point.getYInt(), point.getXInt()), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
						   color=colorText, thickness=1, fontScale=0.6)

	def line_with_tag(self, pointA, pointB, text, colorLine=clr.to_rgba('blue'), colorText=clr.to_rgba('red')):
		self.img = cv2.line(self.img, (pointA.getYInt(), pointA.getXInt()), (pointB.getYInt(), pointB.getXInt()), color=colorLine,
					   thickness=1)
		midp = pointA.getMidPoint(pointB)
		self.img = cv2.putText(img=self.img, text=text, org=(midp.getYInt(), midp.getXInt()), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
						  color=colorText, thickness=1, fontScale=0.6)

	def distance(self, points, maxDistance, tag=True):
		dim = len(points)
		for i in range(dim):
			for j in range(i + 1, dim):
				# dist = getAbsoluteDistance(points[i], points[j]) perche' sono scambiate la x e la y? non esiste motivo logico
				dist = points[i].getAbsoluteDistance(points[j])
				if dist <= float(maxDistance):
					if tag:
						self.line_with_tag(points[i], points[j], str(int(dist)))
					else:
						self.line(points[i], points[j])

	def info_about_the_closest(self, points, maxDistance):
		dim = len(points)
		for i in range(dim):
			closest = points[i].getClosestPoint(points[:i] + points[i + 1:])
			dist = points[i].getAbsoluteDistance(closest)
			self.tag(points[i], str(int(dist)))

	def star(self, radius, color):
		if self.stars_centers:
			center_point = Point(self.stars_centers[len(self.stars_centers) - 1].x + 3*radius, 3*radius)
		else:
			center_point = Point(3*radius, 3*radius)
		self.stars_centers.append(center_point)
		center = (center_point.x, center_point.y)
		for i in range(radius):
			self.img = cv2.circle(self.img, center, i, color)

	def stars(self, num_stars, color, text="ALERT!"):
		heigth, _, _ = self.img.shape
		radius = int(heigth/80)
		self.tag(Point(7*radius, 2*radius), text=text, colorText=color)
		for i in range(num_stars):
			self.star(radius, color)


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


def find_feet_clusters_dbscan(feet_coords, color=None, draw=None):
	# DBSCAN non accetta liste vuote, quindi se questa lo è esco
	if not feet_coords:
		return [], []

	labels = list(DBSCAN(eps=35, min_samples=2).fit(feet_coords).labels_)

	feet_clusters = {el: [] for el in set(labels)}

	for i in range(len(labels)):
		feet_clusters[labels[i]].append(feet_coords[i])

	if color is not None and draw is not None:
		print("Feet clusters:", feet_clusters)

	try:
		people_coords = list(feet_clusters[-1])
	except KeyError:  # se non ci sono punti sparsi
		people_coords = list()

	for label in feet_clusters:
		if label >= 0:
			if len(feet_clusters[label]) > 2:
				print("In questo cluster ci sono", len(feet_clusters[label]), "punti:", feet_clusters[label])

			p = Point.createFromList(feet_clusters[label][0]).getMidPoint(Point.createFromList(feet_clusters[label][1]))
			people_coords.append([p.getXInt(), p.getYInt()])

			if color is not None and draw is not None:
				draw.points([p], radius=3, colorPoints=color)

	return feet_clusters, people_coords


def find_people_clusters_dbscan(people_coords, colors=None, draw=None):
	# DBSCAN non accetta liste vuote, quindi se questa lo è esco
	if not people_coords:
		return []

	labels = list(DBSCAN(eps=80, min_samples=2).fit(people_coords).labels_)

	people_clusters = {el: [] for el in set(labels)}

	for i in range(len(labels)):
		people_clusters[labels[i]].append(people_coords[i])

	if colors is not None and draw is not None:
		print("People clusters:", people_clusters)
		i = 0
		for label in people_clusters:
			if label >= 0:
				color = clr.to_rgba(colors[i % len(colors)])
				draw.points(people_clusters[label], radius=3, colorPoints=color)
				for persona in people_clusters[label]:
					draw.tag(Point.createFromList([persona[0]+25, persona[1]-10]), str(i+1), colorText=clr.to_rgba('yellow'))
				i += 1

	return people_clusters


def find_near_keypoints(keypoint_coords, hidden_depth=None):
	# metto tutti i kp in un'unico array piatto e di interi, lo giro di DBSCAN che mi dice quali punti sono vicini.
	# poi vedo se ci sono punti di persone diverse che appartengono allo stesso cluster. Nel caso considero le due
	# persone vicine
	num_persone = len(keypoint_coords)
	all_kp = keypoint_coords.copy()

	# per ogni persona potrei guardare quale tra leftAnkle (15) e rightAnkle (caviglia, 16) è più confident, e poi
	# ottenere le coordinate di quella parte. Qui prendo direttamente rightAnkle e guardo il valore della depth
	# nella matrice hidden_depth e lo inserisco in tutte le coordinate dei punti di quella persona
	if hidden_depth is not None:
		depth_column = []
		for persona in keypoint_coords:
			coord_ankle = findNearest(persona[16])
			point_depth = hidden_depth[coord_ankle[0]][coord_ankle[1]] * 10
			depth_column.append([[point_depth] for _ in range(17)])
		all_kp = np.append(all_kp, depth_column, axis=2)

	all_kp = all_kp.reshape([num_persone * 17, 2 if hidden_depth is None else 3])

	if args.showplt:
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(all_kp[:, 2], all_kp[:, 1] * -1, all_kp[:, 0] * -1, s=60)
		ax.view_init(azim=200)
		plt.show()

	labels = list(DBSCAN(eps=30, min_samples=2).fit(all_kp).labels_)

	# lo raggruppo nuovamente per persona

	labels = np.array(labels).reshape((num_persone, 17))

	# per ogni persona vedo quali label ha

	cluster_nelle_persone = {}
	for i, persona_label in enumerate(labels):
		cluster_nelle_persone[i] = set(persona_label)

	# confronto le varie persone per vedere se hanno label in comune. Se si, contrassegno quel cluster come cluster
	# interpersonale e quindi da evidenziare

	clusters_interpersonali = []
	persone_vicine = []  # array di (p1, p2, cluster_id)

	for i in cluster_nelle_persone.keys():
		for j in range(i+1, len(cluster_nelle_persone)):
			cluster_comune = list(cluster_nelle_persone[i].intersection(cluster_nelle_persone[j]))
			if len(cluster_comune) > 1 or (-1 not in cluster_comune and len(cluster_comune) > 0):
				# non c'è solo il -1 dei punti sparsi in comune
				# se il cluster non è -1 e non è già in quelli da controllare
				for cluster in cluster_comune:
					if cluster != -1:
						persone_vicine.append((i, j, cluster))
						if cluster not in clusters_interpersonali:
							clusters_interpersonali.append(cluster)

	# così con persone_vicine posso subito vedere quali persone sono vicine tra loro
	# con labels e clusters_interpersonali invece posso disegnare i punti in comune: quando disegno i punti controllo
	# il label corrispondente e se si trova in clusters_interpersonali

	return labels, clusters_interpersonali, persone_vicine

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
	def __init__(self, model_name, save_dir, use_cuda, opt_level, verbose, save_visualisations=True):
		super().__init__(model_name, save_dir, use_cuda, save_visualisations)
		self.save_dir = save_dir
		self.opt_level = opt_level
		self.verbose = verbose
		self.posenet_model = posenet.load_model(args.posenet_model)
		if self.use_cuda:
			self.posenet_model = self.posenet_model.cuda()
		else:
			self.posenet_model = self.posenet_model.cpu()
		self.output_stride = self.posenet_model.output_stride

	def posenet_predict(self, filename, base_out_image=None, hidden_depth=None):
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

			kp_labels, clusters_interpersonali, persone_vicine = find_near_keypoints(keypoint_coords, hidden_depth)

			print("Persone vicine:", persone_vicine)

			draw_image = posenet.draw_skel_and_kp(
				base_out_image, pose_scores, keypoint_scores, keypoint_coords, kp_labels, clusters_interpersonali,
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
		timestamp_manager = Time(save_dir=self.save_dir, img_name=image_path, use_cuda=self.use_cuda, opt_level=self.opt_level)

		print("Predicting for {}".format(image_path))
		original_image, preprocessed_image = self._load_and_preprocess_image(image_path)
		pred = self.model_manager.model(preprocessed_image)
		pred = pred['1/1'].data.cpu().numpy().squeeze(0)

		timestamp_manager.img_shape = original_image.size

		filename, _ = os.path.splitext(os.path.basename(image_path))
		npy_save_path = os.path.join(self.save_dir, "outputs", filename + '.npy')
		print("└> Saving predictions to {}".format(npy_save_path))
		np.save(npy_save_path, pred)

		if self.save_visualisations:
			# print(pred[1].shape, pred.shape)
			# tutti i pred[0 -> 3] hanno shape (256, 448)
			hidden_ground = cv2.resize(pred[1], original_image.size) > 0.95
			print(hidden_ground.shape)

			# STEP TIME
			timestamp_manager.add_step("footprints")
			clusters, numeroCluster, clustersInfo, feet = find_clusters(hidden_ground)
			# STEP TIME
			timestamp_manager.add_step("find_clusters")

			feet = np.expand_dims(feet, axis=2).astype(np.int)
			print(feet.shape)
			hidden_depth = cv2.resize(sigmoid_to_depth(pred[3]), original_image.size)

			# TODO: da qui indentificare il centro di ogni footprint e vedere la distanza nello stesso punto della
			# hidden_depth in modo da vedere quanto è distante nella scena
			original_image = np.array(original_image) / 255.0

			# normalise the relevant parts of the depth map and apply colormap
			_max = hidden_depth[hidden_ground].max()
			_min = hidden_depth[hidden_ground].min()
			hidden_depth_normalized = (hidden_depth - _min) / (_max - _min)
			depth_colourmap = self.colormap(hidden_depth_normalized)[:, :, :3]  # ignore alpha channel

			# create and save visualisation image
			hidden_ground = hidden_ground[:, :, None]
			visualisation_footprints = original_image * (1 - hidden_ground) + depth_colourmap * hidden_ground
			visualisation_depth = original_image * 0.05 + depth_colourmap * 0.95

			# visualisation = original_image * 0.05 + depth_colourmap * 0.95
			# on = np.ones(shape=(682, 1024, 1))
			# off = np.zeros(shape=(682, 1024, 1))
			# colors = np.concatenate((on, off, off), axis=2)
			# visualisation = original_image * (1 - feet) + feet * colors #np.ones(shape=original_image.shape)
			visualisation = original_image * (1 - feet) + feet * depth_colourmap
			print(visualisation.shape)

			draw = Draw(visualisation)
			# trovo i baricentri dei clusters e li associo all'immagine
			points = [Point(clusterInfo.com[0], clusterInfo.com[1]) for clusterInfo in clustersInfo.values() if
							clusterInfo.isFoot]
			draw.points(points, radius=1)

			feet_coords = [[point.getXInt(), point.getYInt()] for point in points]

			# STEP TIME
			timestamp_manager.add_step("colormap+feet_coords")
			feet_clusters, people_coords_dbscan = find_feet_clusters_dbscan(feet_coords, clr.to_rgba('red'), draw)
			# STEP TIME
			timestamp_manager.add_step("find_feet_clusters_dbscan")

			# a partire dai baricentri accoppio i piedi identificando le persone e associo questi punti all'immagine
			peoplePoints = onePointEachPerson(points, 31)  # massima distanza tollerabile tra i piedi
			draw.points(peoplePoints, colorPoints=clr.to_rgba('yellow'))

			colors = ["orange", "green", "blue", "chocolate", "dimgrey", "black"]

			# associo all'immagine le linee che uniscono le persone con tag riferito a distanza
			draw.distance(points=peoplePoints, maxDistance=100)

			# STEP TIME
			timestamp_manager.add_step("peoplePoints")
			people_clusters_dbscan = find_people_clusters_dbscan(people_coords_dbscan, colors, draw)
			# STEP TIME
			timestamp_manager.add_step("find_people_clusters_dbscan")

			# associo all'immagine un tag per ogni persona con scritto la distanza della persona piu vicina
			# visualisation = draw_info_about_the_closest(img=visualisation, points=peoplePoints, maxDistance=100)

			# prova per stelle
			draw.stars(5, clr.to_rgba('red'))

			visualisation = draw.get_img()

			visualisation_footprints = (visualisation_footprints[:, :, ::-1] * 255).astype(np.uint8)
			visualisation_depth = (visualisation_depth[:, :, ::-1] * 255).astype(np.uint8)
			visualisation = (visualisation[:, :, ::-1] * 255).astype(np.uint8)

			# STEP TIME
			timestamp_manager.add_step("image_conversion_for_output")
			visualisation = self.posenet_predict(image_path, visualisation, hidden_depth)
			# STEP TIME
			timestamp_manager.add_step("posenet_predict")

			vis_save_path_footprints = os.path.join(self.save_dir, "visualisations", filename + '_footprints.jpg')
			vis_save_path_depth = os.path.join(self.save_dir, "visualisations", filename + '_depth.jpg')
			vis_save_path = os.path.join(self.save_dir, "visualisations", filename + '.jpg')
			print("└> Saving visualisation to {}".format(vis_save_path))
			cv2.imwrite(vis_save_path_footprints, visualisation_footprints)
			cv2.imwrite(vis_save_path_depth, visualisation_depth)
			cv2.imwrite(vis_save_path, visualisation)

		if self.verbose:
			timestamp_manager.write_info()


def posenet_params(parser: argparse.ArgumentParser):
	parser.add_argument("--posenet_model", type=int, default=101)
	parser.add_argument('--scale_factor', type=float, default=1.0)
	parser.add_argument('--notxt', action='store_true')
	parser.add_argument('--showplt', action='store_true')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--opt_level', type=str, choices=['float64', 'float32', 'float16'], default='float64')


if __name__ == '__main__':
	args = parse_args(posenet_params)

	inference_manager = ObstacleManager(
		model_name=args.model,
		use_cuda=torch.cuda.is_available() and not args.no_cuda,
		opt_level=args.opt_level,
		verbose=args.verbose,
		save_visualisations=not args.no_save_vis,
		save_dir=args.save_dir)
	inference_manager.predict(image_path=args.image)



