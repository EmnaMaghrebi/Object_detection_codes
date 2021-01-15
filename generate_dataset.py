"""
Generate dataset.
"""
import os
import numpy as np
from cv2 import imwrite, cvtColor, COLOR_RGB2BGR
import cv2
from pyrender import OffscreenRenderer, RenderFlags, Viewer, Mesh
from PIL import Image
from cameraCharacteristics import CameraPose, CameraDistance
import csv 
from transformations import *

def trim_values(labels, cutoff=7):
    """We trim the values before writing to CSV file
    
    cutoff is the maximum number of digits to include for each entry in the labels
    """

    # Trim values if needed
    labels = list(map(str, labels))
    for j in range(len(labels)):
        value = labels[j]
        if value[0] == '-':
            actual_cutoff = cutoff+2
        else:
            actual_cutoff = cutoff+1
        labels[j] = value[:actual_cutoff] if len(value) > actual_cutoff else value
    return labels

def angleGenerator(label_set, cam_poses, distance, cam, scene, iterations=10, 
                   mode=CameraPose.TOP, fix_angle=0, skip_default_view=False, 
                   force_t_z=-500, force=False, at_x=0.0, at_y=0.0, at_z=0.0):
    """Generates cameras with different perspectives. 
    If mode is TOP, the cameras start from looking straight down into flower with an alpha angle equal to fix_angle
    and move closer to horizontal plane by modifying beta angle (with iterations number of perspectives).
    If mode is SIDE, the cameras start from looking at the flower from the side with a beta angle equal to fix_angle
    and move in a circle around the flower by modifying the alpha angle (with iterations number of perspectives).
    
    skip_default_view is used when we generate calls this method to create a sort of dome of perspective and 
    we don't want duplicate views from directly above the flower (but since alpha angle varies we may also want them 
    since they aren't necessarily duplicate views?).

    The new cameras and labels are added the cam_poses and label_set respectively.
    """
    assert iterations > 0
    assert mode == CameraPose.TOP or mode == CameraPose.SIDE

    # If mode is TOP: 1 x unit_angle = 90 degrees / iterations
    if mode == CameraPose.TOP:
        # If we are looking at a point that is above the z = 0 plane
        # we can take some viewing angles to be from "underneath"
        # the point we are looking at
        if at_z > 0:
            # Depending on the mesh we might want to reach a view from 
            # further down. For example: unit_angle = np.pi / (2 * iterations)
            # would mean that we start from top view (directly above flower and 
            # looking down) and finish by viewing it from the side (camera will 
            # be parallel to z = 0 plane). If unit angle = 2 * np.pi / (3 * iterations)
            # camera will do same as before but will continue down until it reaches
            # (2/3) * pi as the beta angle

            # Two angle presets that have been used
            # unit_angle = 3 * np.pi / (4 * iterations)
            unit_angle = 2 * np.pi / (3 * iterations)

        # Otherwise we don't go below the z = at_z plane 
        else: 
            unit_angle = np.pi / (2 * iterations)

    # If mode is SIDE: 1 x unit_angle = 360 degrees / iterations
    else:
        unit_angle = 2 * np.pi / iterations 

    # Skip first (default view) if skip_default_view = True
    angle_range = range(1,iterations) if skip_default_view else range(iterations)

    for i in angle_range:

        angle = i * unit_angle

        # If mode = TOP: beta angle varies and alpha is fixed
        if mode == CameraPose.TOP:
            alpha = fix_angle
            beta = angle        
        # If mode = SIDE: alpha angle varies and beta is fixed
        else:
            alpha = angle
            beta = fix_angle  

        new_campose, labels = lookAt(distance=distance, alpha=alpha, beta=beta, 
                                     force_t_z=force_t_z, force=force, at_x=at_x, at_y=at_y, at_z=at_z)

        # Add a camera with new pose
        cam_poses.append(new_campose)
        label_set.append(labels)
    
def generate(scene, renderer, flags, alpha, beta, camera, iterations, flower_pose, meshes,
             skip_sub_mesh, at_x=0, at_y=0, at_z=0, force_t_z=-500, skip_default_view=False, offset=0,
             force=False, distance=CameraDistance.MEDIUM, mode=CameraPose.TOP, 
             csvfile='flowers_dataset.csv', csvMode='a', flower_name='', salt=''):
    """Generates flower images and their labels.
    Calls angleGenerator and creates a dome of camera views if mode = NONE, creates side views (incrementing 
    alpha angle and keeping beta fix) if mode = SIDE and creates a top to horizontal view (incrementing beta 
    angle and keeping alpha fix) if mode = TOP.

    If image is specified each view of the flower is inserted into the image (which becomes a background) and
    the resulting image is written to disk with name "background###.jpg". 
    The image has to be of dimensions 1280 x 1280 but this can be changed.

    For each image generated, an associated image with bounding boxes around the flower and center (if they are
    visible) is generated with "BB" as prefix to the original filename.

    The csv file to save the labels to can be specified. By default, the file is opened in append mode but it can
    be opened in write ('w') mode to overwrite any preexisting data.

    flower_pose is used to compute bounding box and the 3D position to 2D pixel mapping
    """
    assert iterations > 0
    
    # Lists that will contain all the camera nodes and their respective labels 
    label_set = []
    cam_poses = []

    # If NONE create dome
    if mode == CameraPose.NONE:
        # We iterate over different alpha angles (i.e we turn around the vertial axis)
        # and for each alpha angle we start from a top view and move our way down to a side view.
        unit_angle = 2 * np.pi / iterations
        for i in range(iterations):
            skip_default_v = i ^ 0 if skip_default_view else False
            angleGenerator(label_set, cam_poses, distance,camera, scene, iterations, 
                           mode=CameraPose.TOP, fix_angle=i*unit_angle, skip_default_view=skip_default_v, 
                           force_t_z=force_t_z, force=force, at_x=at_x, at_y=at_y, at_z=at_z)
    else:
        # If SIDE we fix beta and turn around the vertical axis
        if mode == CameraPose.SIDE:
            fix_angle = beta
        # If TOP we fix alpha and move gradually from a "top" view
        # down until we are at a side view
        else:
            fix_angle = alpha

        angleGenerator(label_set, cam_poses, distance,camera, scene, iterations, mode=mode, 
        fix_angle=fix_angle, force_t_z=force_t_z, force=force, at_x=at_x, at_y=at_y, at_z=at_z)

    print("Generated poses.")

    with open(csvfile,csvMode,newline='') as fd:

        writer = csv.writer(fd)
        IMAGE_WIDTH = renderer.viewport_width
        IMAGE_HEIGHT = renderer.viewport_height
        margin = np.zeros((3, IMAGE_WIDTH, 3), dtype=np.uint8)

        if csvMode == 'w':
            # If we are in "write" mode we start by writing the headers/labels
            print("Writing labels.")
            writer.writerow([
                'image_id', # self-explanatory
                'xtl','ytl','xbr','ybr', # flower 2D bounding box in pixel coordinates
                'target'
                ])

        for cam_idx, cam_pose in enumerate(cam_poses):

            cam_node = scene.add(camera, pose=cam_pose)
            inv_cam_pose = np.linalg.inv(cam_pose)

            # Collections to write to csv file
            center_centroid_2D = []
            bounding_box_flower_2D = []
            target = []
            bounding_boxes_3D = {'flower':[], 'center':[], 'stem':[]}
            bounding_boxes_3D_in_2D = {'flower':[], 'center':[], 'stem':[]}

            # Render scene
            color, _ = renderer.render(scene, flags=flags)
            color = cvtColor(color, COLOR_RGB2BGR)
            box_3d_viz = color.copy()
            box_2d_viz = color.copy()

            for (mesh,name) in meshes:

                if (skip_sub_mesh[name]):
                    # If we skip the stem we don't do anything more than save zeros in
                    # the csv file

                    # Save 2D pixels of 3D bounding box in list 
                    bounding_boxes_3D_in_2D[name] = [0] * 16

                    # Save 3D position of bounding box in list 
                    bounding_boxes_3D[name] = [0] * 24

                else:
                
                    bounds = mesh.bounds

                    min_x = np.min(bounds[0, 0])
                    max_x = np.max(bounds[1, 0])
                    min_y = np.min(bounds[0, 1])
                    max_y = np.max(bounds[1, 1])
                    min_z = np.min(bounds[0, 2])
                    max_z = np.max(bounds[1, 2])

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

                    homogenous_points_3d = np.concatenate([bounding_box_3d, np.ones((bounding_box_3d.shape[0], 1))], axis=1)
                    homogenous_points_3d = flower_pose @ homogenous_points_3d.T

                    # If a point (usually of the stem) lies below the z = 0 plane
                    # that point is projected onto said plane such that all points
                    # for every every mesh have a z coordinate >= 0
                    # This piece of code is not really used on the more recent meshes
                    # since we disregard the stem.
                    if (np.any(homogenous_points_3d[2,:] < 0)):
                        for i in range(0, homogenous_points_3d.shape[1], 2):
                            col_A = homogenous_points_3d[:,i].ravel()
                            col_B = homogenous_points_3d[:,i+1].ravel() 
                            x_A, x_B = col_A[0], col_B[0] 
                            y_A, y_B = col_A[1], col_B[1]
                            z_A, z_B = col_A[2], col_B[2]
                            if (z_A < 0 or z_B < 0):
                                phi = - z_A / (z_B - z_A)
                                new_x = phi * (x_B - x_A) + x_A
                                new_y = phi * (y_B - y_A) + y_A
                                new_z = 0
                                new_col = np.array([new_x, new_y, new_z, 1.0])
                                if z_A < 0:
                                    col_A = new_col
                                else:
                                    col_B = new_col
                            homogenous_points_3d[:,i] = col_A
                            homogenous_points_3d[:,i+1] = col_B

                    camera_proj = camera.get_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT)
                    points_2d = np.matmul(camera_proj, np.matmul(inv_cam_pose, homogenous_points_3d)).T

                    points_2d[:, 0] = (points_2d[:, 0] / points_2d[:, 3]) * IMAGE_WIDTH / 2 + IMAGE_WIDTH / 2
                    points_2d[:, 1] = (points_2d[:, 1] / -points_2d[:, 3]) * IMAGE_HEIGHT / 2 + IMAGE_HEIGHT / 2
                    points_2d = points_2d.astype(np.int32)
                    points_2d = points_2d[:,:2]

                    # Save 2D pixels of 3D bounding box in list 
                    point_coords_for_mesh_2D = []
                    for point in points_2d:
                        point_coords_for_mesh_2D += point.tolist()
                    bounding_boxes_3D_in_2D[name] = point_coords_for_mesh_2D

                    # Save 3D position of bounding box in list 
                    point_coords_for_mesh_3D = []
                    for point in homogenous_points_3d.T[:,:-1]:
                        point_coords_for_mesh_3D += point.tolist()
                    bounding_boxes_3D[name] = point_coords_for_mesh_3D
                
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

                # Add the centroid of the center mesh
                if name == 'center':
                    centroid = np.hstack((mesh.centroid, np.array([1]))).T
                    centroid_3D = ((flower_pose @  centroid).T)
                    centroid_2D = np.matmul(camera_proj, np.matmul(inv_cam_pose, centroid_3D)).T
                    centroid_2D[0] = (centroid_2D[0] / centroid_2D[3]) * IMAGE_WIDTH / 2 + IMAGE_WIDTH / 2
                    centroid_2D[1] = (centroid_2D[1] / -centroid_2D[3]) * IMAGE_HEIGHT / 2 + IMAGE_HEIGHT / 2
                    centroid_2D = centroid_2D.astype(np.int32)                   
                    box_3d_viz = cv2.circle(box_3d_viz, (centroid_2D[0], centroid_2D[1]), radius=5, thickness=3, color=(0,0,0))

                    # Save list to write to csv file
                    center_centroid_2D = [centroid_2D[0], centroid_2D[1]]

                ############# GENERATE 2D BOX OF THE PROJECTED MESH #############
                elif name == 'flower':
                    
                    points_3d = mesh.primitives[0].positions

                    homogenous_points_3d = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], 1)
                    homogenous_points_3d = flower_pose @ homogenous_points_3d.T

                    camera_proj = camera.get_projection_matrix(IMAGE_WIDTH, IMAGE_HEIGHT)
                    points_2d = np.matmul(camera_proj, np.matmul(inv_cam_pose, homogenous_points_3d)).T

                    points_2d[:, 0] = (points_2d[:, 0] / points_2d[:, 3]) * IMAGE_WIDTH / 2 + IMAGE_WIDTH / 2
                    points_2d[:, 1] = (points_2d[:, 1] / -points_2d[:, 3]) * IMAGE_HEIGHT / 2 + IMAGE_HEIGHT / 2
                    points_2d = points_2d.astype(np.int32)

                    min_x = np.min(points_2d[:, 0])
                    max_x = np.max(points_2d[:, 0])
                    min_y = np.min(points_2d[:, 1])
                    max_y = np.max(points_2d[:, 1])

                    # Save coordinates to write to csv file
                    bounding_box_flower_2D = [min_x, max_x, min_y, max_y]
                    target = [1]

                    TL = (min_x, min_y) # ----> x
                    TR = (max_x, min_y)#  !
                    BR = (max_x, max_y)#  !
                    BL = (min_x, max_y)#  y

                    box_2d_viz = cv2.circle(box_2d_viz, TL, radius=5, thickness=3, color=(0,0,0))
                    box_2d_viz = cv2.circle(box_2d_viz, TR, radius=5, thickness=3, color=(0,0,0))
                    box_2d_viz = cv2.circle(box_2d_viz, BR, radius=5, thickness=3, color=(0,0,0))
                    box_2d_viz = cv2.circle(box_2d_viz, BL, radius=5, thickness=3, color=(0,0,0))


                    box_2d_viz = cv2.line(box_2d_viz, TL, TR, color=to_plot[2], thickness=3)
                    box_2d_viz = cv2.line(box_2d_viz, TR, BR, color=to_plot[2], thickness=3)
                    box_2d_viz = cv2.line(box_2d_viz, BR, BL, color=to_plot[2], thickness=3)
                    box_2d_viz = cv2.line(box_2d_viz, BL, TL, color=to_plot[2], thickness=3)

                    bounding_box_flower_2D = [min_x, max_x, min_y, max_y]

                #####################

            file_name = flower_name+'_'+salt+'_'+str(cam_idx + offset)
            #cv2.imwrite('./outputs/compact/com_'+file_name+'.png', np.vstack([color, margin, box_3d_viz, margin, box_2d_viz]))
            cv2.imwrite('./outputs/detection/det_'+file_name+'.png', box_2d_viz)
            #cv2.imwrite('./outputs/pose/pos_'+file_name+'.png', box_3d_viz)
            cv2.imwrite('./outputs/natural/nat_'+file_name+'.png', color)

            # Write values to csv file
            writer.writerow(
                ['nat_'+file_name] +  
                bounding_box_flower_2D +
                target
                )

            scene.remove_node(cam_node)

            if (cam_idx+1) % 5 == 0:
                print("\r%d images out of %d done." %(cam_idx+1, len(cam_poses)), end='')