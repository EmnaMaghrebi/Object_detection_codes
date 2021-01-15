import os
# switch to "osmesa" or "egl" before loading pyrender
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
import cv2

Y_FOV = np.pi / 3.0
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 400
s = np.sqrt(2)/2


for cam_idx, camera_pose in enumerate([
	np.array([
	   [0.0, -s,   s,   0.2],
	   [1.0,  0.0, 0.0, 0.0],
	   [0.0,  s,   s,   0.6],
	   [0.0,  0.0, 0.0, 1.0],
	]),
	np.array([
	   [0.0, -s,   s,   0.35],
	   [1.0,  0.0, 0.0, 0.0],
	   [0.0,  s,   s,   0.40],
	   [0.0,  0.0, 0.0, 1.0],
	]),
	np.array([
	   [0.0, -s,   s,   0.33],
	   [1.0,  0.0, 0.0, 0.0],
	   [0.0,  s,   s,   0.33],
	   [0.0,  0.0, 0.0, 1.0],
	])

]):

	MESH_FP = 'examples/models/fuze.obj'

	fuze_trimesh = trimesh.load(MESH_FP)
	mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
	scene = pyrender.Scene()
	scene.add(mesh)
	camera = pyrender.PerspectiveCamera(yfov=Y_FOV, aspectRatio=IMAGE_WIDTH/IMAGE_HEIGHT)

	scene.add(camera, pose=camera_pose)
	light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
	                           innerConeAngle=np.pi/16.0,
	                           outerConeAngle=np.pi/6.0)
	scene.add(light, pose=camera_pose)
	r = pyrender.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
	print(scene)
	color, _ = r.render(scene)



	############# GENERATE 2D PROJECTION OF THE 3D BOUNDING BOX #############
	box_3d_viz = color.copy()

	points_3d = mesh.primitives[0].positions

	min_x = np.min(points_3d[:, 0])
	max_x = np.max(points_3d[:, 0])
	min_y = np.min(points_3d[:, 1])
	max_y = np.max(points_3d[:, 1])
	min_z = np.min(points_3d[:, 2])
	max_z = np.max(points_3d[:, 2])

	bounding_box_3d = np.array([
			[min_x, min_y, min_z],
			[min_x, min_y, max_z],
			[min_x, max_y, min_z],
			[min_x, max_y, max_z],
			[max_x, min_y, min_z],
			[max_x, min_y, max_z],
			[max_x, max_y, min_z],
			[max_x, max_y, max_z],
	])

	camera_proj = camera.get_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT)
	homogenous_points_3d = np.concatenate([bounding_box_3d, np.ones((bounding_box_3d.shape[0], 1))], 1)
	points_2d = np.matmul(camera_proj, np.matmul(np.linalg.inv(camera_pose), homogenous_points_3d.T)).T


	points_2d[:, 0] = (points_2d[:, 0] / points_2d[:, 3]) * IMAGE_WIDTH / 2 + IMAGE_WIDTH / 2
	points_2d[:, 1] = (points_2d[:, 1] / -points_2d[:, 3]) * IMAGE_HEIGHT / 2 + IMAGE_HEIGHT / 2
	points_2d = points_2d.astype(np.int32)

	for point in points_2d:
		box_3d_viz = cv2.circle(box_3d_viz, (point[0], point[1]), radius=5, thickness=3, color=(0,0,0))


	for to_plot in [
			(0, 1, (255, 0, 0)), 
			(0, 2, (0, 0, 255)),
			(0, 4, (0, 255, 0)), 
			(1, 3, (0, 0, 255)),
			(1, 5, (0, 255, 0)),
			(2, 3, (255, 0, 0)),
			(2, 6, (0, 255, 0)),
			(3, 7, (0, 255, 0)), 
			(4, 5, (255, 0, 0)),
			(4, 6, (0, 0, 255)),
			(5, 7, (0, 0, 255)), 
			(6, 7, (255, 0, 0))]:
			
			
		p1 = (int(points_2d[to_plot[0],0]), int(points_2d[to_plot[0],1]))
		p2 = (int(points_2d[to_plot[1],0]), int(points_2d[to_plot[1],1]))

		box_3d_viz = cv2.line(box_3d_viz, p1, p2, color=to_plot[2], thickness=3)

	############# GENERATE 2D BOX OF THE PROJECTED MESH #############
	box_2d_viz = color.copy()

	points_3d = mesh.primitives[0].positions

	camera_proj = camera.get_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT)
	homogenous_points_3d = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], 1)
	points_2d = np.matmul(camera_proj, np.matmul(np.linalg.inv(camera_pose), homogenous_points_3d.T)).T


	points_2d[:, 0] = (points_2d[:, 0] / points_2d[:, 3]) * IMAGE_WIDTH / 2 + IMAGE_WIDTH / 2
	points_2d[:, 1] = (points_2d[:, 1] / -points_2d[:, 3]) * IMAGE_HEIGHT / 2 + IMAGE_HEIGHT / 2
	points_2d = points_2d.astype(np.int32)

	min_x = np.min(points_2d[:, 0])
	max_x = np.max(points_2d[:, 0])
	min_y = np.min(points_2d[:, 1])
	max_y = np.max(points_2d[:, 1])

	TL = (min_x, min_y)
	TR = (max_x, min_y)
	BR = (max_x, max_y)
	BL = (min_x, max_y)

	box_2d_viz = cv2.circle(box_2d_viz, TL, radius=5, thickness=3, color=(0,0,0))
	box_2d_viz = cv2.circle(box_2d_viz, TR, radius=5, thickness=3, color=(0,0,0))
	box_2d_viz = cv2.circle(box_2d_viz, BR, radius=5, thickness=3, color=(0,0,0))
	box_2d_viz = cv2.circle(box_2d_viz, BL, radius=5, thickness=3, color=(0,0,0))


	box_2d_viz = cv2.line(box_2d_viz, TL, TR, color=to_plot[2], thickness=3)
	box_2d_viz = cv2.line(box_2d_viz, TR, BR, color=to_plot[2], thickness=3)
	box_2d_viz = cv2.line(box_2d_viz, BR, BL, color=to_plot[2], thickness=3)
	box_2d_viz = cv2.line(box_2d_viz, BL, TL, color=to_plot[2], thickness=3)
	#####################

	margin = np.zeros((3, IMAGE_WIDTH, 3), dtype=np.uint8)
	cv2.imwrite('viz_'+str(cam_idx)+'.png', np.vstack([color, margin, box_3d_viz, margin, box_2d_viz]))




