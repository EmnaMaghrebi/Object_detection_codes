from transformations import spatial_transform_Matrix
import numpy as np

def get_flower_pose(name='', file_name=''):
    """There is a predefined pose for each flower type, and sometimes even for each
    mesh within a flower type
    """
    if name == '':
        return spatial_transform_Matrix()
    elif name == 'gerbera':
        return spatial_transform_Matrix(scale=0.0002, roll=0, yaw=0, t_z=-0.02)
    elif name ==  'poppy_anemone':
        return spatial_transform_Matrix(scale=0.002, yaw = np.pi/6, roll=np.pi/24, t_z=-0.33)
    elif name ==  'capitata':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=0, t_z=0.0001)
    elif name ==  'onion':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=0, t_z=-0.05)
    elif name ==  'celery':
        return spatial_transform_Matrix(scale=0.4, yaw = 0, roll=0, t_z=-0.15)
    elif name ==  'italica':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=0, t_z=0.0001)
    elif name ==  'cauliflower':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=0, t_z=0.0001)
    elif name ==  'chili_pepper':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=0, pitch=-np.pi/4, t_z=0.0001)
    elif name ==  'watermelon':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=np.pi/6, t_z=0.0001)
    elif name ==  'cantaloupe':
        if file_name == 'VG09_6.obj':
            return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=np.pi/9, pitch=np.pi/4, t_z=0.0001)
        elif file_name == 'VG09_9.obj':
            return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=np.pi/6, pitch=np.pi/9, t_z=0.0001, t_x=-0.3, t_y=0.3)
    elif name ==  'cucumber':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=np.pi/6, t_z=0.0001)
    elif name ==  'cucurbita_maxima':
        return spatial_transform_Matrix(scale=0.3, yaw = 0, pitch=-np.pi/9, t_z=0.0001)
    elif name ==  'zucchini':
        if file_name == 'VG12_6.obj':
            return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=np.pi/5, t_z=0.0001)
        elif file_name == 'VG12_7.obj':
            return spatial_transform_Matrix(scale=0.3, yaw = 0, roll=-np.pi/12, pitch=np.pi/4, t_z=0.0001)
    elif name ==  'sunflower':
        return spatial_transform_Matrix(scale=0.3, yaw=-np.pi, roll=-np.pi/2.5, pitch=0, t_z=-0.3)
    elif name ==  'bean':
        return spatial_transform_Matrix(scale=0.3, yaw=np.pi/2, roll=-np.pi/9, pitch=-np.pi/1.5, t_z=-0.15)
    elif name ==  'peas':
        return spatial_transform_Matrix(scale=0.3, yaw=-np.pi/12, roll=np.pi/1.8, pitch=0, t_z=-0.2)
    elif name ==  'cherry_tomato':
        return spatial_transform_Matrix(scale=0.3, pitch=np.pi/3.12, t_z=-0.1)
    elif name ==  'eggplant':
        return spatial_transform_Matrix(scale=0.3, roll=-np.pi/1.8, yaw=np.pi, t_z=-0.1)
    elif name ==  'tea':
        return spatial_transform_Matrix(scale=0.3, roll=np.pi/5, pitch=np.pi/5, t_z=0)  
    elif name ==  'strawberry':
        if file_name == 'AG06_5.obj':
            return spatial_transform_Matrix(scale=0.3, roll=np.pi/2.5, yaw=-np.pi/12, t_z=0)   
        elif file_name == 'AG06_9.obj':
            return spatial_transform_Matrix(scale=0.3, roll=np.pi/5, yaw=0, t_z=0) 
    elif name ==  'cotton':
        if file_name == 'AG08_5.obj':
            return spatial_transform_Matrix(scale=0.3, roll=np.pi/5, pitch=-np.pi/12, t_z=0)   
        elif file_name == 'AG08_6.obj':
            return spatial_transform_Matrix(scale=0.3, roll=np.pi/4, pitch=np.pi/6, t_z=0)   
    elif name ==  'tobacco':
        if file_name == 'AG13_7.obj':
            return spatial_transform_Matrix(scale=0.3, roll=-np.pi/3.5, yaw=np.pi, t_z=-0.12)   
        else:
            return spatial_transform_Matrix(scale=0.3, roll=np.pi/4, pitch=0, t_z=0)   
    elif name == 'crocus':
        if file_name == 'FL43_1.glb' or file_name == 'FL43_6.glb' or file_name == 'FL43_7.glb':
            return spatial_transform_Matrix(scale=1, roll=0, pitch=0, t_z=-0.03) 
        elif file_name == 'FL43_2.glb':
            return spatial_transform_Matrix(scale=1, roll=-np.pi/5, pitch=0, t_z=-0.03) 
    elif name == 'dahlia':
        if file_name == 'FL44_1.glb':
            return spatial_transform_Matrix(scale=0.4, roll=-np.pi/16, pitch=np.pi/12, t_z=-0.1) 
        elif file_name == 'FL44_9.glb':
            return spatial_transform_Matrix(scale=0.4, roll=0, pitch=np.pi/12, t_z=-0.1) 
    elif name == 'gardenia':
        return spatial_transform_Matrix(scale=0.4, roll=-np.pi/6, pitch=np.pi/12, yaw=np.pi, t_z=-0.03) 
    elif name == 'spider_gerbera':
        if file_name == 'FL46_6.glb':
            return spatial_transform_Matrix(scale=0.3, roll=0, pitch=-np.pi/12, yaw=0, t_z=-0.1) 
        elif file_name == 'FL46_7.glb':
            return spatial_transform_Matrix(scale=0.3, roll=0, pitch=np.pi/12, yaw=0, t_z=-0.1) 
        elif file_name == 'FL46_9.glb':
            return spatial_transform_Matrix(scale=0.3, roll=-np.pi/8, pitch=0, yaw=0, t_z=-0.07) 
        else:
            return spatial_transform_Matrix(scale=0.3, roll=0, pitch=np.pi/12, yaw=0, t_z=-0.06) 
    elif name == 'hibiscus':
        if file_name == 'FL47_1.glb':
            return spatial_transform_Matrix(scale=0.3, roll=0, pitch=np.pi/8, yaw=0, t_z=-0.2) 
        elif file_name == 'FL47_4.glb':
            return spatial_transform_Matrix(scale=0.3, roll=0, pitch=-np.pi/8, yaw=0, t_z=-0.25) 
        elif file_name == 'FL47_7.glb':
            return spatial_transform_Matrix(scale=0.3, roll=np.pi/9, pitch=np.pi/10, yaw=0, t_z=-0.15) 
    elif name == 'lotus':
        return spatial_transform_Matrix(scale=0.2, roll=0, pitch=0, yaw=0, t_z=-0.1)
    elif name == 'orchid':
        return spatial_transform_Matrix(scale=0.2, roll=-np.pi/2, pitch=0, yaw=np.pi, t_z=0)  
    elif name == 'peony':
        if file_name == 'FL51_3.glb':
            return spatial_transform_Matrix(scale=0.2, roll=0, pitch=-np.pi/8, yaw=0, t_z=0)  
        if file_name == 'FL51_8.glb':
            return spatial_transform_Matrix(scale=0.2, roll=0, pitch=np.pi/6, yaw=0, t_z=0)  
    elif name == 'passionflower':
        if file_name == 'FL52_1.glb':
            return spatial_transform_Matrix(scale=0.2, roll=-np.pi/2.5, pitch=0, yaw=np.pi, t_z=-0.03)  
        elif file_name == 'FL52_5.glb':
            return spatial_transform_Matrix(scale=0.2, roll=-np.pi/8, pitch=-np.pi/10, yaw=np.pi, t_z=0)  
        elif file_name == 'FL52_6.glb':
            return spatial_transform_Matrix(scale=0.2, roll=np.pi/5, pitch=-np.pi/8, yaw=0, t_z=-0.05)  
    elif name == 'african_violet':
        if file_name == 'FL54_1.glb' or file_name == 'FL54_4.glb' or file_name == 'FL54_5.glb':
            return spatial_transform_Matrix(scale=0.2, roll=np.pi/20, pitch=np.pi/20, yaw=0, t_z=0)  
        elif file_name == 'FL54_7.glb':
            return spatial_transform_Matrix(scale=0.2, roll=-np.pi/10, pitch=-np.pi/8, yaw=np.pi, t_z=0)  
    elif name == 'bat_flower':
        return spatial_transform_Matrix(scale=0.2, roll=np.pi/2, pitch=0, yaw=0, t_z=0)  
    elif name == 'marigold':
        return spatial_transform_Matrix(scale=0.8, roll=-np.pi/6, pitch=0, yaw=np.pi, t_z=0)  


# We group together the meshes that have their flower, center and stem mesh in the
# same order
set_1 = ['FL32_2.obj']
set_2 = ['FL22_1.obj', 'FL22_3.obj', 'FL22_4.obj']
set_3 = ['FL22_2.obj', 'FL22_5.obj']
set_4 = ['FL32_3.obj', 'FL57_6.glb']
set_5 = ['VG05_3.obj', 'VG05_4.obj', 'VG05_7.obj', 'VG04_2.obj']
set_6 = ['FL32_1.obj']
set_7 = ['VG01_7.obj']
set_8 = ['VG01_8.obj', 'VG01_9.obj']
set_9 = ['VG02_8.obj']
set_10 = ['VG06_4.obj', 'AG13_7.obj']
set_11 = ['VG07_9.obj']
set_12 = ['VG08_2.obj']
set_13 = ['VG09_6.obj', 'VG09_9.obj', 'FL49_2.glb', 'FL49_7.glb']
set_14 = ['VG11_3.obj', 'AG02_9.obj', 'FL50_9.glb']
set_15 = ['VG12_6.obj', 'VG12_7.obj']
set_16 = ['VG14_4.obj']
set_17 = ['VG16_6.obj']
set_18 = ['VG17_6.obj', 'AG06_5.obj']
set_19 = ['VG18_1.obj', 'FL56_1.glb', 'VG10_3.obj']
set_20 = ['VG19_3.obj']
set_21 = ['AG06_9.obj', 'FL47_7.glb', 'FL54_1.glb']
set_22 = ['AG08_5.obj', 'AG08_6.obj', 'FL57_5.glb']
set_23 = ['FL43_1.glb']
set_24 = ['FL43_2.glb', 'FL43_6.glb', 'FL43_7.glb']
set_25 = ['FL44_1.glb']
set_26 = ['FL44_9.glb', 'FL51_3.glb', 'FL51_8.glb']
set_27 = ['FL45_2.glb', 'FL57_2.glb']
set_28 = ['FL46_1.glb', 'FL46_4.glb']
set_29 = ['FL46_6.glb', 'FL46_7.glb']
set_30 = ['FL46_9.glb']
set_31 = ['FL47_1.glb', 'FL47_4.glb']
set_32 = ['FL49_4.glb']
set_33 = ['FL50_4.glb']
set_34 = ['FL52_1.glb']
set_35 = ['FL52_6.glb']
set_36 = ['FL52_5.glb']
set_37 = ['FL54_4.glb', 'FL54_7.glb']
set_38 = ['FL54_5.glb']

def get_flower_mesh_indices(file_name):
    """We retrieve the indices for the flower, center and stem sub-meshes
    of a given mesh

    In the more recent meshes used the stem index is arbitrary since
    that part of the flower isn't used.
    """
    index1, index2, index3 = 0, 0, 0
    if file_name in set_1:
        index1, index2, index3 = 1, 1, 2
    elif file_name in set_2:
        index1, index2, index3 = 0, 1, 3
    elif file_name in set_3:
        index1, index2, index3 = 0, 0, 3
    elif file_name in set_4:
        index1, index2, index3 = 2, 0, 2
    elif file_name in set_5:
        index1, index2, index3 = 0, 0, 0
    elif file_name in set_6:
        index1, index2, index3 = 0, 0, 1
    elif file_name in set_7:
        index1, index2, index3 = 3, 3, 3
    elif file_name in set_8:
        index1, index2, index3 = 6, 3, 5
    elif file_name in set_9:
        index1, index2, index3 = 5, 0, 5
    # elif file_name in set_10:
    #     index1, index2, index3 = 1, 0, 1
    elif file_name in set_10:
        index1, index2, index3 = 0, 0, 0
    elif file_name in set_11:
        index1, index2, index3 = 2, 3, 5
    elif file_name in set_12:
        # index1, index2, index3 = 4, 1, 0
        index1, index2, index3 = 2, 2, 1
    elif file_name in set_13:
        index1, index2, index3 = 4, 0, 0
    elif file_name in set_14:
        index1, index2, index3 = 1, 1, 0
    elif file_name in set_15:
        index1, index2, index3 = 3, 2, 0
    elif file_name in set_16:
        index1, index2, index3 = 4, 2, 0
    elif file_name in set_17:
        index1, index2, index3 = 2, 2, 0
    elif file_name in set_18:
        index1, index2, index3 = 4, 3, 0
    elif file_name in set_19:
        index1, index2, index3 = 3, 3, 0
    elif file_name in set_20:
        index1, index2, index3 = 3, 1, 0
    elif file_name in set_21:
        index1, index2, index3 = 3, 2, 0
    elif file_name in set_22:
        index1, index2, index3 = 1, 0, 0
    elif file_name in set_23:
        index1, index2, index3 = 7, 7, 0 
    elif file_name in set_24:
        index1, index2, index3 = 9, 8, 0    
    elif file_name in set_25:
        index1, index2, index3 = 0, 3, 0
    elif file_name in set_26:
        index1, index2, index3 = 0, 1, 0    
    elif file_name in set_27:
        index1, index2, index3 = 2, 1, 0     
    elif file_name in set_28:
        index1, index2, index3 = 2, 6, 0  
    elif file_name in set_29:
        index1, index2, index3 = 4, 2, 0   
    elif file_name in set_30:
        index1, index2, index3 = 10, 3, 0    
    elif file_name in set_31:
        index1, index2, index3 = 5, 3, 0  
    elif file_name in set_32:
        index1, index2, index3 = 5, 1, 0
    elif file_name in set_33:
        index1, index2, index3 = 6, 6, 0
    elif file_name in set_34:
        index1, index2, index3 = 4, 4, 0
    elif file_name in set_35:
        index1, index2, index3 = 1, 6, 0
    elif file_name in set_36:
        index1, index2, index3 = 3, 6, 4
    elif file_name in set_37:
        index1, index2, index3 = 3, 4, 0
    elif file_name in set_38:
        index1, index2, index3 = 5, 4, 0
    return index1, index2, index3