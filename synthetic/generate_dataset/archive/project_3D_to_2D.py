"""
Test mapping 3D to 2D
"""
import os
import numpy as np
import trimesh
from cv2 import cvtColor,\
                COLOR_RGB2BGR,\
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
# Scene creation
#==============================================================================

scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

#==============================================================================
# Adding objects to the scene
#==============================================================================

# Flower
flower_mesh = mesh_list.pop(index_1)
flower_node = scene.add(flower_mesh, pose=flower_pose, name='flower')

# Center (aka stigma)
center_mesh = mesh_list.pop(index_2)
center_node = scene.add(center_mesh, pose=flower_pose, name='center')

# The other sub-meshes of the flowers
for i in range(len(mesh_list)):
    scene.add(mesh_list[i], pose=flower_pose, name='flower'+str(i))

# Base square, inf square and walls
base_node = scene.add(base_mesh, pose=base_pose)
inf_base_node = scene.add(inf_base_mesh, pose=inf_base_pose)
for pose in wall_poses:
    scene.add(wall_mesh, pose=pose, name='wall')

# To add a white sphere at the position of the stigma (center) of the flower
# center_point = ((flower_pose @  np.concatenate((center_mesh.centroid, np.array([1]))).T).T)
# center_point = center_point.astype('float32')
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
# 3D to 2D mapping
#==============================================================================

# camera properties
yfov = (np.pi / 3.0)
height = 600
width = 600
fov = 1.0 / (np.tan(yfov/2.0))
aspect_ratio = width / height
zfar = 10

# camera
cam = PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio, zfar=zfar)

# distance to near and far clipping planes
far = cam.zfar
near = cam.znear

# Specify parameters of camera of scene that will be previewed
# alpha, beta, distance, force_height, force = 0, np.pi/4, 0.1, 0.35, False
alpha, beta, distance, force_height, force = np.pi/6, np.pi/3, 0.3, 0, False
new_campose, labels = generateCameraPoseFromSpherical(distance=distance, alpha=alpha, beta=beta, force_t_z=force_height)
cam_node = scene.add(cam, pose=new_campose)


# There are different definitions of the clip matrix based on if 
# the system is row major or column major, not sure which one
# is correct..?

def clip_matrix__(fov, aspect_ratio, far, near):
    return np.array([
        [fov * aspect_ratio,   0,                       0, 0],
        [                 0, fov,                       0, 0],
        [                 0,   0,   (far+near)/(far-near), 1],
        [                 0,   0, (2*near*far)/(near-far), 0]
    ])

def clip_matrix_(yfov, aspect_ratio, far, near):
    return np.array([
        [ 1 / (aspect_ratio * np.tan(yfov/2.0)),                    0,                     0,                       0],
        [                                     0, 1 / np.tan(yfov/2.0),                     0,                       0],
        [                                     0,                    0, (far+near)/(near-far), (2*near*far)/(near-far)],
        [                                     0,                    0,                    -1,                       0]
    ])

def clip_matrix(fov, aspect_ratio, far, near):
    return np.array([
        [fov * aspect_ratio,   0,                     0,                       0],
        [                 0, fov,                     0,                       0],
        [                 0,   0, (far+near)/(far-near), (2*near*far)/(near-far)],
        [                 0,   0,                     1,                       0]
    ])

# All matrices are (4x4) and vectors are (4,)

# centroid of the center of the flower before applying the position transformation
centroid = np.hstack((center_mesh.centroid, np.array([1]))).T

# centroid_3D is the position (x,y,z) of the centroid in the coordinate system
# centroid_3D = [x, y, z, 1]
centroid_3D = ((flower_pose @  centroid).T)

# inverse camera matrix
# Maybe have to play around with transposing certain matrices
# i.e. "invcam = np.linalg.inv(new_campose.T)"
invcam = np.linalg.inv(new_campose)

# Different possibilities for the clip matrix
clip_matrix = clip_matrix__(fov, aspect_ratio, far, near)
# clip_matrix = clip_matrix_(yfoc, aspect_ratio, far, near)
# clip_matrix = clip_matrix(fov, aspect_ratio, far, near)


# Seems to be different ways to obtain the 2D points...?
# point_2d = clip_matrix @ invcam @ centroid_3D

# This version sometimes produces (x,y) values that are inside the
# bounds of the image (0 <= x < width and 0 <= y < height)
# but they are not on the center (stigma) of the flower
# Again, not sure if we should take the transpose of invcam or not
invcenter = flower_pose @ invcam.T @ centroid
point_2d = clip_matrix @ invcenter


# Divide first 3 elements of the vector by the last one
point_2d[:-1] /= point_2d[-1]

x = point_2d[0]
y = point_2d[1]
w = point_2d[3]

# compute pixels
new_x = (x * width ) / (2.0 * w) + width/2.0
new_y = (y * height) / (2.0 * w) + height/2.0

# another way to compute the pixels..?
# new_x = (x / w ) *  width
# new_y = (y / w ) *  height

print("Pixel coordinates: (%f,%f)" %(new_x, new_y))
points = np.array([[new_x, new_y]], dtype='int')
print(points)


# Not used (Sutherland-Hodgeman algorithm)
def clip(subjectPolygon, clipPolygon):
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
   return(outputList)

# View the scene
view_render_flags = {'cull_faces': False}
v = Viewer(scene, viewport_size = (width, height),render_flags=view_render_flags)
r = OffscreenRenderer(viewport_width=width, viewport_height=height)
color, depth = r.render(scene, flags=RenderFlags.SKIP_CULL_FACES)
# color = cvtColor(color, COLOR_RGB2BGR)

def draw_point(img, imgpts):
    for point in imgpts:
        img = circle(img, tuple(point), 4, (255,0,0), 2)    
    return img

# draw the computed (x,y) coords on the image
color = draw_point(color, points)
plt.imshow(color)
plt.show()