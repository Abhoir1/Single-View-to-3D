import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 

import imageio

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)

def get_mesh_renderer(image_size=512, lights=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def render_vox(vox, output_path):
    device=vox.device
    mesh = pytorch3d.ops.cubify(vox, thresh=0.5, device=device)
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0))
    textures = textures * torch.tensor([0.7, 0.7, 1], device=device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)

    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    lights = pytorch3d.renderer.PointLights(location=[[3.0, 0, 0]]).to(device)
    renderer = get_mesh_renderer(image_size=512, device=device)
    
    num_frames = 60
    fps = 15
    
    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []
    
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
    for elevation, azimuth in zip(elevations, azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=2.0,
            elev=elevation,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()
        rend_uint8 = (rend * 255).clip(0, 255).astype(np.uint8)
        images.append(rend_uint8)
        
    imageio.mimsave(output_path, images, fps=fps, loop=0)

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def render_pointcloud(src_point, output_path, image_size=512, num_samples=200, device='cuda'):

    src_point = src_point[0].detach()
    points = src_point
    color = (points - points.min()) / (points.max() - points.min())
    color = color.to(torch.float32)  # Ensure color is of type torch.cuda.FloatTensor

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)

    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(torus_point_cloud, cameras=cameras)

    num_frames = 60
    fps = 15
    # output_path = "point_video.gif"
    
    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    for elevation, azimuth in zip(elevations, azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=1.0,
            elev=elevation,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)
        rend = renderer(torus_point_cloud, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy() 
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)

    imageio.mimsave(output_path, images, fps=fps, loop=0)

def render_mesh(mesh, output_path):
    device=mesh.device
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0), device = device)
    textures = textures * torch.tensor([0.7, 0.7, 1], device = device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    # render_360_mesh(mesh.detach(), file_name=output_path, device=device)

    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    lights = pytorch3d.renderer.PointLights(location=[[3.0, 0, 0]]).to(device)
    renderer = get_mesh_renderer(image_size=512, device=device)
    
    num_frames = 60
    fps = 15
    
    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []
    
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
    for elevation, azimuth in zip(elevations, azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=1.0,
            elev=elevation,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()
        rend_uint8 = (rend * 255).clip(0, 255).astype(np.uint8)
        images.append(rend_uint8)
        
    imageio.mimsave(output_path, images, fps=fps, loop=0)


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis_freq', default=1000, type=str)
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


# def get_mesh_renderer(image_size=512, lights=None, device=None):
#     """
#     Returns a Pytorch3D Mesh Renderer.

#     Args:
#         image_size (int): The rendered image size.
#         lights: A default Pytorch3D lights object.
#         device (torch.device): The torch device to use (CPU or GPU). If not specified,
#             will automatically use GPU if available, otherwise CPU.
#     """
#     if device is None:
#         if torch.cuda.is_available():
#             device = torch.device("cuda:0")
#         else:
#             device = torch.device("cpu")
#     raster_settings = RasterizationSettings(
#         image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
#     )
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(raster_settings=raster_settings),
#         shader=HardPhongShader(device=device, lights=lights),
#     )
#     return renderer


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

import numpy as np

# def render_and_save_voxel(voxel, file_name, args,  device = 'cuda'):
#     R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)

#     voxel_size = 32
#     # voxel = voxel[0].cpu()
#     # voxel = voxel.numpy()

#     renderer = get_mesh_renderer(image_size=512, device=device)

#     min_value = -1
#     max_value = 1

#     if isinstance(voxel, torch.Tensor):
#         voxel = voxel.cpu().numpy()

#     vertices, faces = mcubes.marching_cubes(voxel, isovalue=0)

#     vertices = torch.tensor(vertices).float()
#     faces = torch.tensor(faces.astype(int))

#     vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
#     textures = ((vertices - vertices.min()) / (vertices.max() - vertices.min()))

#     textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))
#     mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

#     num_frames = 60
#     fps = 15
#     # output_path = "voxelgrid.gif"

#     elevations = torch.linspace(0, 360, num_frames, device=device)
#     azimuths = torch.linspace(0, 360, num_frames, device=device)
#     images = []
    
#     lights = pytorch3d.renderer.PointLights(location=[[3.0, 0, 0]]).to(device)
    
#     cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

#     for elevation, azimuth in zip(elevations, azimuths):
#         R, T = pytorch3d.renderer.look_at_view_transform(
#             dist=2.0,
#             elev=elevation,
#             azim=azimuth
#         )
#         cameras.R = R.to(device)
#         cameras.T = T.to(device)

#         rend = renderer(mesh, cameras=cameras, lights=lights)
#         rend = rend[0, ..., :3].cpu().numpy() 
#         rend_uint8 = (rend * 255).clip(0, 255).astype(np.uint8)
#         images.append(rend_uint8)

#     imageio.mimsave(file_name, images, fps=fps, loop=0)


def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])

        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics

def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
        
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    # max_iter = 20
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        # TODO:
        # if (step % args.vis_freq) == 0:
        #     # visualization block
        #     #  rend = 
        #     plt.imsave(f'vis/{step}_{args.type}.png', rend)

        if step % 50 == 0:

            if args.type == "vox":
                render_vox(predictions[0], output_path=f'Visualization/Visual_vox/Predicted_{step}.gif')
                vox_gt = feed_dict['voxels'].to(args.device)
                render_vox(vox_gt[0], output_path=f'Visualization/Visual_vox/GroundTruth_{step}.gif')

            if args.type == "point":
                pred_points = predictions.cpu()
                render_pointcloud(pred_points, f'Visualization/Visual_point/Predicted_{step}.gif')
                gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
                render_pointcloud(gt_points, f'Visualization/Visual_point/GroundTruth_{step}.gif')

            if args.type == "mesh":
                render_mesh(predictions.detach(), output_path=f'Visualization/Visual_mesh/Predicted_{step}.gif')
                render_mesh(mesh_gt.to(args.device), output_path=f'Visualization/Visual_mesh/GroundTruth_{step}.gif')

                
        # if (step % args.vis_freq) == 0:
        #     # Visualize and save the RGB image, ground truth voxel, and predicted voxel.
        #     # Save input RGB
        #     # plt.imsave(f'vis/{step}_input_rgb.png', images_gt.squeeze().cpu())
        #     image_tensor = images_gt.squeeze().cpu().numpy()

        #     # Optional: Normalize to [0,1] if values are in [0,255]
        #     # image_tensor = image_tensor / 255.0

        #     plt.imsave(f'vis/{step}_input_rgb.png', image_tensor)
            
        #     # Save Ground Truth Voxel
        #     gt_voxel = feed_dict['voxels'].squeeze().cpu().numpy()
        #     render_and_save_voxel(gt_voxel, f'vis/{step}_gt_voxel.png', args)
            
        #     # Save Predicted Voxel
        #     pred_voxel = predictions.squeeze().detach()
        #     render_and_save_voxel(pred_voxel, f'vis/{step}_pred_voxel.png', args)

      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    
    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
