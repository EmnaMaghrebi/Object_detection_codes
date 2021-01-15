"""
Test projectPoints from OpenCV
"""
import os
import numpy as np
import trimesh
from cv2 import imwrite,\
                cvtColor,\
                COLOR_RGB2BGR,\
                COLOR_BGR2GRAY,\
                Rodrigues,\
                projectPoints,\
                imread,\
                findChessboardCorners,\
                cornerSubPix,\
                waitKey,\
                drawChessboardCorners,\
                destroyAllWindows,\
                TERM_CRITERIA_EPS,\
                TERM_CRITERIA_MAX_ITER,\
                imshow,\
                calibrateCamera,\
                solvePnP,\
                line,\
                circle
from pyrender import PerspectiveCamera, IntrinsicsCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags

from PIL import Image
from cameraCharacteristics import CameraPose, CameraDistance
import math
import csv 
from mesh_util import as_mesh
from transformations import *
import matplotlib.pyplot as plt
from generate_dataset import generate,\
                             generate_chess_images
from flower_utils import get_flower_pose,\
                         get_flower_mesh_indices
import glob

# A few colors
yellowRGB = [236, 228, 53]
orchidRGB = [153,50,204]
lettuceGreenRGB = [2, 173, 7]
redRGB = [255, 0, 0]

#==============================================================================
# Mesh creation
#==============================================================================

file_to_type = {'FL22_1.obj': 'poppy_anemone', 'FL22_2.obj': 'poppy_anemone',
'FL22_3.obj': 'poppy_anemone', 'FL22_4.obj': 'poppy_anemone', 'FL22_5.obj': 'poppy_anemone',
'FL32_1.obj': 'gerbera', 'FL32_2.obj': 'gerbera', 'FL32_3.obj': 'gerbera'}

# User specifies filename
file_name = 'FL22_5.obj'

# Get the flower_type based on the filename
flower_type = file_to_type[file_name]

# Determine path name
path_name = './models/flower_models/obj/'+flower_type+'/'

# Load the obj file
flower_trimesh = trimesh.load(path_name+file_name)

if (type(flower_trimesh) == trimesh.scene.scene.Scene):
    trimesh_list = flower_trimesh.dump()
else:
    trimesh_list = [flower_trimesh]

# Convert from Trimesh to Mesh
mesh_list = []
for tri_mesh in trimesh_list:
    mesh_list.append(Mesh.from_trimesh(tri_mesh))

# Get indices of flower_mesh and center_mesh in the mesh_list
index_1, index_2 = get_flower_mesh_indices(file_name)

# Different options for the base
base_types = ['soil', 'plain', 'other']
base_type = base_types[0]

# Base mesh
base_trimesh = trimesh.load('./models/scene/'+base_type+'_base.obj')
base_mesh = Mesh.from_trimesh(base_trimesh)
base_pose = spatial_transform_Matrix(scale = 1/2, t_z=-0.01)

# Inf base mesh (big square that sits underneath the scene)
inf_base_trimesh = trimesh.load('./models/scene/'+base_type+'_base.obj')
inf_base_mesh = Mesh.from_trimesh(base_trimesh)
inf_base_pose = spatial_transform_Matrix(scale = 4, t_z=-1)

# Different options for the walls
wall_types = ['hydroponic_farm', 'other']
wall_type = wall_types[0]

# Side "walls" mesh
wall_trimesh = trimesh.load('./models/scene/'+wall_type+'_wall.obj')
wall_mesh = Mesh.from_trimesh(wall_trimesh)

#==============================================================================
# Light creation
#==============================================================================

# Lights for the inf_base and walls
side_intensity = 3.0
inf_base_dir_l = DirectionalLight(color=np.ones(3), intensity=10.0)
spot_l_sides = PointLight(color=np.ones(3), intensity=side_intensity)

# Light for the flower
point_l = PointLight(color=np.ones(3), intensity=1.0)

#==============================================================================
# Poses
#==============================================================================

# Side wall poses
side_closeness = 1
side_height = 0

side_E_pose = spatial_transform_Matrix(roll=np.pi/2, yaw=np.pi/2, t_x=-1/side_closeness, t_z=side_height)
side_N_pose = spatial_transform_Matrix(roll=np.pi/2, t_y=1/side_closeness, t_z=side_height)
side_W_pose = spatial_transform_Matrix(roll=np.pi/2, yaw=-np.pi/2, t_x=1/side_closeness, t_z=side_height)
side_S_pose = spatial_transform_Matrix(roll=np.pi/2, yaw=np.pi, t_y=-1/side_closeness, t_z=side_height)
wall_poses = [side_E_pose, side_N_pose, side_W_pose, side_S_pose]

# Poses for lights for the inf_base and walls
inf_base_dir_pose, ignore = generateCameraPoseFromSpherical(view=CameraPose.TOP, force_t_z=-1/2, force=True)
point_l_E = spatial_transform_Matrix(roll=np.pi/12, yaw=np.pi/2, t_x=-1/((side_closeness)+0.5),t_z=side_height+1/8)
point_l_N = spatial_transform_Matrix(roll=np.pi/12, t_y=1/((side_closeness)+0.5),t_z=side_height+1/8)
point_l_W = spatial_transform_Matrix(roll=np.pi/12, yaw=-np.pi/2, t_x=1/((side_closeness)+0.5),t_z=side_height+1/8)
point_l_S = spatial_transform_Matrix(roll=np.pi/12, yaw=np.pi, t_y=-1/((side_closeness)+0.5),t_z=side_height+1/8)
wall_light_poses = [point_l_E, point_l_N, point_l_W, point_l_S]

# Pose for light on flower
light_pose, ignore = generateCameraPoseFromSpherical(view=CameraPose.SIDE, distance=CameraDistance.MEDIUM, force_t_z=0.2, force=True)

flower_pose = get_flower_pose(flower_type)

#==============================================================================
# Camera creation
#==============================================================================

cam = PerspectiveCamera(yfov=(np.pi / 3.0))

#==============================================================================
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

#==============================================================================
# Adding objects to the scene
#==============================================================================

# We don't need to add the flower in this script

# Flower
flower_mesh = mesh_list.pop(index_1)
# flower_node = scene.add(flower_mesh, pose=flower_pose, name='flower')

# # Center (aka stigma)
center_mesh = mesh_list.pop(index_2)
# center_node = scene.add(center_mesh, pose=flower_pose, name='center')

# # The other sub-meshes of the flowers
# for i in range(len(mesh_list)):
#     scene.add(mesh_list[i], pose=flower_pose, name='flower'+str(i))

# Base square, inf square and walls
base_node = scene.add(base_mesh, pose=base_pose)
inf_base_node = scene.add(inf_base_mesh, pose=inf_base_pose)
for pose in wall_poses:
    scene.add(wall_mesh, pose=pose, name='wall')

center_point = ((flower_pose @  np.concatenate((center_mesh.centroid, np.array([1]))).T).T)[:-1]
center_point = center_point.astype('float32')

# print(((flower_pose @  np.concatenate(center_mesh.centroid, np.array([[1]])).T).T))
# ball_trimesh = trimesh.creation.icosphere(subdivisions=3)
# ball_mesh = Mesh.from_trimesh(ball_trimesh)
# scene.add(ball_mesh, pose=spatial_transform_Matrix(scale=0.005, t_x=center_point[0], t_y=center_point[1], t_z=center_point[2]))

#==============================================================================
# Adding lights
#==============================================================================

point_l_node = scene.add(point_l, pose=light_pose)
inf_base_dire_node = scene.add(inf_base_dir_l, pose=inf_base_dir_pose)

for wall_light_pose in wall_light_poses:
    scene.add(spot_l_sides, pose=wall_light_pose, name='wall_light')

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================

# Specify parameters of camera of scene that will be previewed
# alpha, beta, distance, force_height, force = 0, np.pi/4, 0.1, 0.35, False
alpha, beta, distance, force_height, force = 0, np.pi/12, 0.2, 0, False

new_campose, labels = generateCameraPoseFromSpherical(distance=distance, alpha=alpha, beta=beta, force_t_z=force_height)
cam_node = scene.add(cam, pose=new_campose)

#==============================================================================
# Create chess board for projection of points
#==============================================================================

chess_board_trimesh = trimesh.load('./models/scene/chess_board.obj')
chess_board_mesh = Mesh.from_trimesh(chess_board_trimesh)
chess_board_pose = spatial_transform_Matrix(scale = distance/3, yaw=0, roll=0)
# chess_board_pose = spatial_transform_Matrix(scale = distance/3, yaw=np.pi/4, roll=np.pi/12, t_z = 0.09)

chess_board_node = scene.add(chess_board_mesh, pose=chess_board_pose, name='chess_board')

# Number of points
num_points = 9

# Generate grid of (x,y,z=0) points lying on the chessboard
# Hard to understand how points are mapped to the points that are
# detected, so obj_points_numpy is overwritten @ line 300
x_coord = np.linspace(-1, 1, num_points)
XX, YY = np.meshgrid(x_coord, x_coord)
number_of_points = (XX.shape[0]-2) * (XX.shape[1]-2)
points_on_plane = np.zeros((number_of_points,4))
for i in range(1, XX.shape[0]-1):
    for j in range(1, XX.shape[1]-1):
        points_on_plane[(i - 1) * (XX.shape[1]-2) + j - 1, :] = np.array([XX[i,j], YY[i,j], 0, 1])

obj_points_numpy = ((chess_board_pose @ points_on_plane.T).T)[:,:-1]
obj_points_numpy = np.where(abs(obj_points_numpy) < 1.0e-6, 0, obj_points_numpy) # Values close to 0 are 0
obj_points_numpy = obj_points_numpy.astype('float32')

# Add spheres at the object points in obj_spoints
# ball_trimesh = trimesh.creation.icosphere(subdivisions=3)
# ball_mesh = Mesh.from_trimesh(ball_trimesh)
# for row in obj_points:
#     scene.add(ball_mesh, pose=spatial_transform_Matrix(scale=0.005, t_x=row[0], t_y=row[1], t_z=row[2]))

#==============================================================================
# Attempt to convert 3D to 2D position
#==============================================================================

# View the scene
view_render_flags = {'cull_faces': False}
v = Viewer(scene, render_flags=view_render_flags)

#==============================================================================
# Rendering offscreen from that camera
#==============================================================================

dimensions = 600
# dimensions = 1280
r = OffscreenRenderer(viewport_width=dimensions, viewport_height=dimensions)

# Generate dataset ?
question_string = "Generate chess board dataset? (select no because it's already there) [y/n]: "
choice = input(question_string)
accepted_inputs = ['y','n']
while (not (choice in accepted_inputs)):
    choice = input(question_string)

# Parameters
# alpha = 0
# beta = np.pi/8
iterations = 20
# distance = 0.7
# force_height = 0.0
mode = CameraPose.SIDE

if choice == 'y':

    # Remove camera that was used as preview from the scene
    scene.remove_node(cam_node)

    generate_chess_images(
        scene=scene, renderer=r, alpha=alpha, beta=beta, cam=cam, 
        iterations=iterations, force_t_z=force_height, force=force, 
        distance=distance, mode=mode, skip_default_view=False, stri='side')

print("Finished")

question_string = "Terminate to inspect dataset? (select no) [y/n]: "
choice = input(question_string)
accepted_inputs = ['y','n']
while (not (choice in accepted_inputs)):
    choice = input(question_string)

if choice == 'n':

    # termination criteria
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images = glob.glob('./outputs/chess_board/*.jpg')

    # Overwrite obj_points_numpy
    obj_points_numpy = np.zeros((6*7, 3), np.float32)
    obj_points_numpy[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    img_points = [] # 2d points on image plane
    obj_points = [] # 3d point in real world space

    for fname in images:
        img = imread(fname)
        gray = cvtColor(img, COLOR_BGR2GRAY) 
        # Find the chess board corners
        ret, corners = findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Found chess board!")
            obj_points.append(obj_points_numpy)
            corners2 = cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            img_points.append(corners)
            # Draw and display the corners
            drawChessboardCorners(img, (7,7), corners2, ret)
            # fname = fname.replace('./outputs/chess_board/','')
            # imwrite('./outputs/chess_board/img'+fname, img)

    img = imread(images[0])
    gray = cvtColor(img, COLOR_BGR2GRAY) 
    ret, mtx, dist, rvecs, tvecs = calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # axis = np.array([[3,0,0], [0,3,0], [0,0,-3]], dtype=float).reshape(-1,3)
    axis = np.array([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    axis = axis.astype('float32')

    # Attempt to find the center point, fails
    # axis = center_point

    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

    def draw_point(img, imgpts):
        for point in imgpts:
            img = circle(img, tuple(point.ravel()), 4, (255,0,0), 2)    
        return img

    for fname in glob.glob('./outputs/chess_board/*.jpg'):
        img = imread(fname)
        gray = cvtColor(img,COLOR_BGR2GRAY)
        ret, corners = findChessboardCorners(gray, (7,6), None)
        if ret == True:
            print("Found again!")
            corners2 = cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = solvePnP(obj_points_numpy, corners2, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = projectPoints(axis, rvecs, tvecs, mtx, dist)
            # print(imgpts)
            imgpts = imgpts.astype('int')
            img = draw(img,corners2,imgpts)
            # img = draw_point(img, imgpts)
            # imshow('img',img)
            fname = fname.replace('./outputs/chess_board/','')
            imwrite('./outputs/chess_board/attempt_1/img'+fname, img)
            # imwrite(fname[:6]+'.png', img)
    print("Result images can be found in: ./outputs/chess_board/attempt_1")
