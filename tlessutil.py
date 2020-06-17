
# Given a random obj id, I need to find a scene with said object.
# Therefore I need a dictionary that, given an obj id, returns a scene id.
# I couldn't find any list of which scenes contains which objects, so I'll have 
# to scrape the gt.yml files.
import os
import ruamel.yaml as yaml
import numpy as np
import pickle 
import os.path
from os import path
import open3d as o3d

# I know, global vars are ugly... There's probably some nicer way to do this.
obj2scenepath = '/home/grans/Documents/prnet2/obj2scenelist.pkl'
scene2objpath = '/home/grans/Documents/prnet2/scene2objlist.pkl'
tless_path='/home/grans/Documents/t-less_v2/'

if not path.exists(obj2scenepath):
    print("The path to the pickled dictionary 'obj2scenelist' doesn't seem to exists. You can create this by running 'create_dicts' and then updating the global parameter in 'tlessutil.py'.")
if not path.exists(scene2objpath):
    print("The path to the pickled dictionary 'scene2objlist' doesn't seem to exists. You can create this by running 'create_dicts' and then updating the global parameter in 'tlessutil.py'.")


def get_random_view(scene_id = None, view_id = None, occlusion_threshold = None):
    # Returns a random triple of scene, view, obj ids.
    # 
    # Occlusion threshold [0.0, 1.0] - Will only pick objects that are visible # with that ratio.
    
    global obj2scenepath, scene2objpath, tless_path

    obj2scene = pickle.load(open(obj2scenepath, 'rb'))
    scene2obj = pickle.load(open(scene2objpath, 'rb'))


    if scene_id is None:
        scene_id = np.random.choice(np.arange(1, 21))
    if view_id is None: 
        view_id = np.random.choice(np.arange(0, 504))
    
    if occlusion_threshold is None:
        obj_id = np.random.choice(scene2obj[scene_id])
    else: 
        obj_areas_dict_path_mask = os.path.join(tless_path, 
                                        'test_primesense',
                                        '{:02d}', 
                                        'obj_areas.pkl')
        obj_areas_dict_path = obj_areas_dict_path_mask.format(scene_id)
        f = open(obj_areas_dict_path, 'rb')
        d = pickle.load(f)
        r = np.array(d[view_id])[:, 0]/np.array(d[view_id])[:, 1]
        obj_id = np.random.choice(np.where(r > occlusion_threshold)[0])
        # print("hold")

    return scene_id, view_id, obj_id

def get_obj_id_in_view(scene_id, view_id):
    # 
    pass

def get_primesense_files(ids, tlesspath='/home/grans/Documents/t-less_v2/'):
    # scene, view, obj
    scene_id, view_id, obj_id = ids
    
    # gt_mask = os.path.join(tlesspath, 'test_primesense', '{:02d}', 'gt.pkl')
    gt_mask = os.path.join(tlesspath, 'test_primesense', '{:02d}', 'gt.yml')
    info_mask = os.path.join(tlesspath, 'test_primesense', '{:02d}', 'info.yml')

    rgb_image_mask = os.path.join(tlesspath, 'test_primesense',
                                        '{:02d}', 'rgb', '{:04d}.png')
    depth_image_mask = os.path.join(tlesspath, 'test_primesense',
                                        '{:02d}', 'depth', '{:04d}.png')

    return rgb_image_mask.format(scene_id, view_id), \
            depth_image_mask.format(scene_id, view_id), \
            gt_mask.format(scene_id), \
            info_mask.format(scene_id)

def get_pcd(ids, zclip=(600,1100)):
    
    scene_id, view_id, obj_id = ids
    rgb_path, depth_path, gt_path, info_path = get_primesense_files(ids)

    z_min = zclip[0]
    z_max = zclip[1]

    f = open(info_path, 'r')
    info = yaml.load(f, Loader=yaml.CLoader)
    
    fx, _, cx, _, fy, cy, _, _, _ = np.array(info[view_id]['cam_K'])
    scale = np.array(info[view_id]['depth_scale'])

    depth_raw = o3d.io.read_image(depth_path)

    # `o3d.geometry.RGBDImage.`create_from_color_and_depth` only allows 
    # max trunc and no min trunc. This is a workaround.
    d = np.asarray(depth_raw).astype(np.float32)
    d *= scale
    mask = np.logical_or(d < z_min, d > z_max)
    d[mask] = 0.0
    depth_raw = o3d.geometry.Image(d)

    width, height = depth_raw.get_max_bound().astype('int')
    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


    color_raw = o3d.io.read_image(rgb_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, 
        depth_raw, 
        depth_scale=(1/scale), 
        depth_trunc=np.inf, 
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        o3d.camera.PinholeCameraIntrinsic(cameraIntrinsics),
        project_valid_depth_only=False
    )

    return pcd

def get_model(model_id, model_type='models_cad_subdivided', tlesspath='/home/grans/Documents/t-less_v2/'):

    if not model_id < 21 and not model_id > 0:
        print("model_id must be in the range of [1, 20]")
        raise ValueError

    if  model_type != 'models_cad' and \
        model_type != 'models_cad_subdivided' and \
        model_type != 'models_reconst':
        print("Supported models types are: 'models_cad', 'models_cad_subdivided', 'models_reconst'")
        raise NotImplementedError

    file_mask = os.path.join(tlesspath, model_type, 'obj_{:02d}.ply')
    mesh = o3d.io.read_triangle_mesh(file_mask.format(model_id))
    return mesh

def get_partial_pcd(ids, window=(400, 400), zclip=(600,1100), vis=False):
    scene_id, view_id, obj_id = ids
    rgb_path, depth_path, gt_path, info_path = get_primesense_files(ids)

    w_w = window[0]
    h_w = window[1]
    z_min = zclip[0]
    z_max = zclip[1]

    f = open(info_path, 'r')
    info = yaml.load(f, Loader=yaml.CLoader)
    # f = open(gt_path, 'rb')
    # gt = pickle.load(f)
    f = open(gt_path, 'r')
    gt = yaml.load(f, Loader=yaml.CLoader)

    model_id = gt[view_id][obj_id]['obj_id']

    fx, _, cx, _, fy, cy, _, _, _ = np.array(info[view_id]['cam_K'])
    scale = np.array(info[view_id]['depth_scale'])

    depth_raw = o3d.io.read_image(depth_path)
    color_raw = o3d.io.read_image(rgb_path)
    width, height = depth_raw.get_max_bound().astype('int')
    
    assert w_w <= width, "Window width is larger than the image."
    assert h_w <= height, "Window height is larger than the image."
    

    # Instead of doing some weird slice thing, I'm simply going to zero out 
    # everything that I don't want in the point cloud. Because Open3D filters
    # out zeros.

    # (x, y, width, height)
    bb = gt[view_id][obj_id]['obj_bb']
    # Let's unpack the tuple for convenience
    x_bb, y_bb, w_bb, h_bb = bb

    R_m2c = np.array(gt[view_id][obj_id]['cam_R_m2c']).reshape(3,3)
    t_m2c = np.array(gt[view_id][obj_id]['cam_t_m2c']).reshape(3,1)

    


    
    # Here I have to do the padding thing, to get a certain "window size"
    # The new x can take values in the range (x_bb -(w_w - w_bb), x_bb)
    # Similarly with y.
    if (w_w - w_bb) < 0 or (h_w - h_bb) < 0:
        print("Warning. Scene: %d View: %d Obj: %d Bbox larger than window." % (scene_id, view_id, obj_id))
        x_w = x_bb
        y_w = y_bb
    else: 
        x_w = np.random.randint(x_bb - (w_w - w_bb), x_bb)
        y_w = np.random.randint(y_bb - (h_w - h_bb), y_bb)

    # In case the window ends up outside of the frame, we simple bump it in.
    if x_w < 0:
        x_w = 0
    if y_w < 0:
        y_w = 0
    if x_w + w_w > width:
        # The window is outside of the image.
        x_w = x_w - ((x_w+w_w) - width)
    if y_w + h_w > height:
        # The window is outside of the image.
        y_w = y_w - ((y_w+h_w) - height)


    # Create a mask based on the the slice
    mask = np.zeros_like(depth_raw)
    # The matrices are indexes as row-column, hence the flip. That is: (y, x)
    mask[y_w:y_w+h_w, x_w:x_w+w_w] = 1

    depth_masked = mask * np.asarray(depth_raw)
    depth_masked = o3d.geometry.Image(depth_masked)

    color_masked = np.repeat(mask[:,:,np.newaxis], 3, axis=2) * np.asarray(color_raw)

    ## Confirming the crop
    if vis:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig,ax = plt.subplots(1)
        # Display the image
        ax.imshow(np.asarray(color_masked))
        # Create a Rectangle patch
        rect = patches.Rectangle((bb[0],bb[1]),bb[2],bb[3],linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()

    
    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, # I think I don't need to mask the color image.
        depth_masked, 
        depth_scale=(1/scale), 
        depth_trunc=np.inf, 
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        o3d.camera.PinholeCameraIntrinsic(cameraIntrinsics),
        project_valid_depth_only=False
    )

    return pcd, model_id, R_m2c, t_m2c

def show_pcd_with_nans(pcd, other_geo=None):
    # If you have a pcd with nan values, Open3D will not visualize them. 
    # This is a workaround.

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    nanmask = np.logical_not(np.isnan(points))
    nanmask = np.any(nanmask, axis=1)
    points = points[nanmask, :]
    colors = colors[nanmask, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if other_geo is not None:
        o3d.visualization.draw_geometries([pcd] + other_geo)
    else:
        o3d.visualization.draw_geometries([pcd])

def pick_points(pcd):
    print("")
    print(
        "1) Pick using [shift + left click]"
    )
    print("   Press [shift + right click] to undo last point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    #vc = vis.get_view_control()
    #vc.set_up([0, -1, 0])#
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()



def create_dicts(tlesspath='/home/grans/Documents/t-less_v2/test_primesense/', save=False, verbose=True): 
    '''
        Creates two dictionaries that relates objects ids to scenes, and vice versa. 

        The first dictionary is 'scene2objlist' which given the scene index, returns a list of all objects present in that scene.
    
        Example: 
        > scene2objectlist[2]
        [5, 6, 7]
        
        > obj2scenelist[1]
        [5, 7, 9, 17, 18, 20]

        Returns: scene2objlist, obj2scenelist 
    '''

    scenes = list(range(1, 21))
    scene2objlist = {scene: None for scene in list(range(1, 21))}
    obj2scenelist = {obj: [] for obj in list(range(1, 31))}

    if verbose:
        print("Creating dictionaries.")
    for scene in scenes:
        gtfile = os.path.join(tlesspath, '{:02d}', 'gt.yml').format(scene)
        with open(gtfile, 'r') as f:
            gts = yaml.load(f, Loader=yaml.CLoader)
            obj_ids = list(np.unique([d['obj_id'] for d in gts[0]]))
            [obj2scenelist[obj_id].append(scene) for obj_id in obj_ids]
            scene2objlist[scene] = obj_ids
            if verbose:
                print(str(scene) + ": " + str(obj_ids))


    if save:
        pickle.dump(scene2objlist, open('scene2objlist.pkl', 'wb'))
        pickle.dump(obj2scenelist, open('obj2scenelist.pkl', 'wb'))

    return scene2objlist, obj2scenelist 

def create_gt_dict(tlesspath='/home/grans/Documents/t-less_v2/test_primesense/', save=False, verbose=True):
    # The way the yaml files are loaded sucks. Instead I convert them to 
    # dictionaries and pickle them.
    # Example: scene would have a gt file. It would be loaded as a list of 
    # views, which is fine. Then each view would be a list of gt for each 
    # object. That sucks, because that number doesn't relate to the actual 
    # obj id.
    # What i want is a list of views consisting of dictionaries.

    # Update: What was i thinking? This ignores multiple instances...
    # Might fix later.

    scenes = list(range(1, 21))
    for scene in scenes:
        gtfile = os.path.join(tlesspath, '{:02d}', 'gt.yml').format(scene)
        pklgtfile = os.path.join(tlesspath, '{:02d}', 'gt.pkl').format(scene)

        print("Creating pickle of scene %02d." % scene)
        
        with open(gtfile, 'r') as f:
            gt = yaml.load(f, Loader=yaml.CLoader)
            for i in range(len(gt)):
                print("%03d, " % i, end='')
                # Convert each list into a dictionary.
                # Use the obj_id as keys.
                keys = [elem['obj_id'] for elem in gt[i]]
                gt[i] = dict(zip(keys, gt[i]))
            print("Scene %02d Done." % scene)
            if save:
                print("Saving to: %s" % pklgtfile)
                f = open(pklgtfile, 'wb')
                pickle.dump(gt, f)


def normalize_pcd(pcd):
    # Normalizes the pcd (in place) by setting the origin to it's center of mass. 
    points = np.asarray(pcd.points)
    # points = points[~np.isnan(points).any(axis=1), :]
    centroid = np.nanmean(points, axis=0)
    pcd = pcd.translate(-centroid)
    return -centroid


if __name__ == "__main__":

    # create_dicts()
    # create_gt_dict(save=True)

    globmax = []
    globmin = []
    # np.random.seed(1234)
    while True:
        ids = get_random_view(occlusion_threshold=0.9)
        # ids = (17, 428, 5)
        print("scene: %d view: %d obj: %d" % ids)
        partial_pcd, model_id, R, t = get_partial_pcd(ids, window=(200, 200))

        T = np.vstack((np.hstack((R, t)), [0, 0, 0, 1]))
        model = get_model(model_id)
        model = model.transform(T)
        model.paint_uniform_color([1, 0.67, 0])
        model.compute_vertex_normals()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)

        t = normalize_pcd(partial_pcd)
        model.translate(t)
        show_pcd_with_nans(partial_pcd, other_geo=[mesh_frame, model])



        # pcd = get_pcd(ids, (-np.inf, np.inf))
        # show_pcd_with_nans(pcd)
        #points = np.asarray(pcd.points)
        #mask = np.all([points[:, 2] > 600, points[:, 2] < 1100], axis=0)
        #savedpoints = points[mask]
        #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #size=10)
        #o3d.visualization.draw_geometries([pcd, mesh_frame_orig, mesh_frame])
        #discardedpoints = points[np.logical_not(mask)]
        
        #pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        #pcd1.paint_uniform_color([0, 1, 0])
        #pcd1.translate([0, 0, -5])

        # pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(discardedpoints))
        # pcd2.paint_uniform_color([1.0, 0, 0])
        # c
        #print(pcd.points)

        # o3d.visualization.draw_geometries([pcd])
        print("hold")
        
        #picked_points = pick_points(pcd)

        # if len(picked_points) == 0:
        #     break
        

        # points = np.asarray(pcd.points)
        # picked = points[picked_points]

        # minz = np.min(points[picked_points, 2])
        # globmin.append(minz)
        # maxz = np.max(points[picked_points, 2])
        # globmax.append(maxz)
        # print("hold")

    print("µMin: %f" % np.mean(globmin))
    print("µMax: %f" % np.mean(globmax))
    print("Hello world.")