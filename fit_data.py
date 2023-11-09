import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
import mcubes
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
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

def fit_mesh(mesh_src, mesh_tgt, args, device):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)

    print('Done!')

    return mesh_src, mesh_tgt

def render_mesh(mesh, output_path, image_size=512, num_samples=200, device=device):
    renderer = get_mesh_renderer(image_size=image_size)

    vertices, faces = mesh.verts_list()[0], mesh.faces_list()[0]

    color = [0.7, 0.7, 1]

    # Create a single color texture for the mesh
    num_vertices = vertices.shape[0]
    colors = torch.tensor(color, device=device).unsqueeze(0)
    colors = colors.expand(num_vertices, -1)

    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    textures = torch.ones_like(vertices, device=device)
    textures = textures * torch.tensor(colors, device=device)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the mesh
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    num_frames = 60
    fps = 15

    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []
    for elevation, azimuth in zip(elevations, azimuths):
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=1.0,
            elev=elevation,
            azim=azimuth
        )
        cameras.R = R.to(device)
        cameras.T = T.to(device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().detach().numpy()
        rend = (rend * 255).astype(np.uint8)
        images.append(rend)

    imageio.mimsave(output_path, images, fps=fps, loop=0)

def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()
    
        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    return voxels_src, voxels_tgt

# def render_voxel(voxels_src,output_path, device=device):
    
#     R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)

#     voxel_size = 32
#     voxels_src = voxels_src[0].detach().cpu()
#     voxels_src = voxels_src.numpy()

#     renderer = get_mesh_renderer(image_size=512, device=device)

#     min_value = -1
#     max_value = 1

#     vertices, faces = mcubes.marching_cubes(voxels_src, isovalue=0)

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

#     imageio.mimsave(output_path, images, fps=fps, loop=0)

# def render_voxel(mesh, file_name, image_size=512, device='cuda'):
#     # Camera position and light settings
#     R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
#     lights = pytorch3d.renderer.PointLights(location=[[3.0, 0, 0]]).to(device)
#     renderer = get_mesh_renderer(image_size=image_size, device=device)
    
#     num_frames = 60
#     fps = 15
    
#     elevations = torch.linspace(0, 360, num_frames, device=device)
#     azimuths = torch.linspace(0, 360, num_frames, device=device)
#     images = []
    
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

def render_voxel(vox, output_path):
    device=vox.device
    mesh = pytorch3d.ops.cubify(vox, thresh=0.5, device=device)
    textures = torch.ones_like(mesh.verts_list()[0].unsqueeze(0))
    textures = textures * torch.tensor([0.7, 0.7, 1], device=device)
    mesh.textures=pytorch3d.renderer.TexturesVertex(textures)
    # render_voxel(mesh, file_name=output_path, device=device)

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

def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    return pointclouds_src, pointclouds_tgt

def render_pointcloud(src_point, output_path, image_size=512, num_samples=200, device=device):

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

def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    feed = r2n2_dataset[0]

    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()

    if args.type == "vox":
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        print("HERE")
        # fit_voxel(voxels_src, voxels_tgt, args)

        voxels_src, voxel_target = fit_voxel(voxels_src, voxels_tgt, args)
    
        # output_path = "voxelgrid.gif"
        
        # render_mesh_from_voxel(voxels_src,  output_path)
        output_path_src2 = "src_voxel.gif"
        output_path_tgt2 = "tgt_voxel.gif"

        render_voxel(voxels_src, output_path_src2)
        render_voxel(voxel_target, output_path_tgt2)

        print("I AM HERE")

        src_gif2 = imageio.mimread(output_path_src2)
        tgt_gif2 = imageio.mimread(output_path_tgt2)

        min_frames2 = min(len(src_gif2), len(tgt_gif2))
        src_gif2 = src_gif2[:min_frames2]
        tgt_gif2 = tgt_gif2[:min_frames2]

        side_by_side_gif2 = [np.hstack((src_frame2, tgt_frame2)) for src_frame2, tgt_frame2 in zip(src_gif2, tgt_gif2)]

        output_path_combined2 = "combined_voxel.gif"
        imageio.mimsave(output_path_combined2, side_by_side_gif2, fps=15, loop=0)


    elif args.type == "point":
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
        pointcloud_src, pointcloud_tgt = fit_pointcloud(pointclouds_src, pointclouds_tgt, args)
        # render_pointcloud(pointcloud_src)

        output_path_src1 = "src_pointcloud.gif"
        output_path_tgt1 = "tgt_pointcloud.gif"

        render_pointcloud(pointcloud_src, output_path_src1)
        render_pointcloud(pointcloud_tgt, output_path_tgt1)

        src_gif1 = imageio.mimread(output_path_src1)
        tgt_gif1 = imageio.mimread(output_path_tgt1)

        min_frames1 = min(len(src_gif1), len(tgt_gif1))
        src_gif1 = src_gif1[:min_frames1]
        tgt_gif1 = tgt_gif1[:min_frames1]

        side_by_side_gif1 = [np.hstack((src_frame1, tgt_frame1)) for src_frame1, tgt_frame1 in zip(src_gif1, tgt_gif1)]

        output_path_combined1 = "combined_pointclouds.gif"
        imageio.mimsave(output_path_combined1, side_by_side_gif1, fps=15, loop=0)

    
    elif args.type == "mesh":
     
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        mesh_src1, tgt_mesh1 = fit_mesh(mesh_src, mesh_tgt, args, device=args.device) 

        output_path_src = "src_mesh.gif"
        output_path_tgt = "tgt_mesh.gif"

        render_mesh(mesh_src1, output_path_src)
        render_mesh(tgt_mesh1, output_path_tgt)

        src_gif = imageio.mimread(output_path_src)
        tgt_gif = imageio.mimread(output_path_tgt)

        min_frames = min(len(src_gif), len(tgt_gif))
        src_gif = src_gif[:min_frames]
        tgt_gif = tgt_gif[:min_frames]

        side_by_side_gif = [np.hstack((src_frame, tgt_frame)) for src_frame, tgt_frame in zip(src_gif, tgt_gif)]

        output_path_combined = "combined_meshes.gif"
        imageio.mimsave(output_path_combined, side_by_side_gif, fps=15, loop=0)

        # render_mesh(mesh_src1, tgt_mesh1)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
