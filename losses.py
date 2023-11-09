import torch
import torch.nn.functional as F
import pytorch3d.loss

# define losses
# def voxel_loss(voxel_src, voxel_tgt):
# 	# voxel_src: b x h x w x d
# 	# voxel_tgt: b x h x w x d
# 	# loss = 
# 	# implement some loss for binary voxel grids
	
#     return loss

def voxel_loss(voxel_src, voxel_tgt):
    # Ensure that both voxel grids have the same shape.
    assert voxel_src.shape == voxel_tgt.shape

    # Apply the sigmoid activation to the source voxel grid.
    voxel_src = torch.sigmoid(voxel_src)

    # Compute the binary cross-entropy loss.
    loss = F.binary_cross_entropy(voxel_src, voxel_tgt)

    return loss

# def voxel_loss(voxel_src,voxel_tgt):
# 	# voxel_src: b x h x w x d
# 	# voxel_tgt: b x h x w x d
# 	sigmoid = torch.nn.Sigmoid()

# 	func = torch.nn.BCELoss() 
# 	loss = func(sigmoid(voxel_src),voxel_tgt)
# 	# implement some loss for binary voxel grids
# 	return loss

# def chamfer_loss(point_cloud_src,point_cloud_tgt):
# 	# point_cloud_src, point_cloud_src: b x n_points x 3  
# 	# loss_chamfer = 
# 	# implement chamfer loss from scratch
# 	loss_chamfer = 0
# 	return loss_chamfer

def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_tgt: b x n_points x 3
    # Initialize the loss to zero
    loss_chamfer = 0

    for i in range(point_cloud_src.size(0)):
        src_points = point_cloud_src[i]
        tgt_points = point_cloud_tgt[i]

        diff = src_points.unsqueeze(1) - tgt_points.unsqueeze(0)  # (n_src, n_tgt, 3)
        dist2 = torch.sum(diff**2, dim=2)  # (n_src, n_tgt)

        min_dist_src = torch.min(dist2, dim=1)[0]  # (n_src,)

        min_dist_tgt = torch.min(dist2, dim=0)[0]  # (n_tgt,)

        loss_chamfer += torch.mean(min_dist_src) + torch.mean(min_dist_tgt)

    loss_chamfer /= point_cloud_src.size(0)

    return loss_chamfer

def smoothness_loss(mesh):
    loss_laplacian = pytorch3d.loss.mesh_laplacian_smoothing(mesh)

    return loss_laplacian
