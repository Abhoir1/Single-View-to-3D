<!-- CMSC848F

Assignment 2: Single View to 3D

Change the root location of dataset as per your system in data_location.py file.
Run the following commands in your terminal to see the results, change the python version as per your system:

1. Exploring loss functions
1.1 Voxel : "python fit_data.py --type 'vox'"        "combined_voxel.gif" will be saved in the root directory along with individual gifs.
1.2 Point : "python fit_data.py --type 'point'"        "combined_point.gif" will be saved in the root directory along with individual gifs.
1.3 Voxel :" python fit_data.py --type 'mesh'"        "combined_mesh.gif" will be saved in the root directory along with individual gifs.

2. Reconstructing 3D from a single view
1.1 "python train_model.py --type 'vox'"           This training will create a checkpoint_vox.pth file in the root directory, use this trained model and evaluate using "python eval_model.py --type 'vox' --load_checkpoint" command. Change which predictions you want to save by changing step, the predictions along with ground truths will be saved in Visualization/Vis_vox directory.

1.2 "python train_model.py --type 'point' "          This training will create a checkpoint_point.pth file in the root directory, use this trained model and evaluate using "python eval_model.py --type 'point' --load_checkpoint" command. Change which predictions you want to save by changing step, the predictions along with ground truths will be saved in Visualization/Vis_point directory.

1.3 "python train_model.py --type 'mesh'  "         This training will create a checkpoint_mesh.pth file in the root directory, use this trained model and evaluate using "python eval_model.py --type 'mesh' --load_checkpoint" command. Change which predictions you want to save by changing step, the predictions along with ground truths will be saved in Visualization/Vis_mesh directory.

After each evalution set the threshold vs F1 score plot is saved in Visualization/Evaluation directory. -->
                    <meta charset="utf-8" emacsmode="-*- markdown -*">
                            **CMSC848F Assignment2: Single View to 3D**

# Setup

Exploring loss functions
===============================================================================

Fitting a Voxel Grid
-------------------------------------------------------------------------------

![Fitted Voxel Grid (left) and Target Voxel Grid (right)](/home/ab/assignment2/vis_fit_data/combined_voxelnew.gif)

Fitting a Point Cloud
-------------------------------------------------------------------------------

![Fitted Point Cloud (left) and Target Point Cloud (right)](/home/ab/assignment2/vis_fit_data/combined_pointclouds.gif)

Fitting a Mesh 
-------------------------------------------------------------------------------

![Fitted Mesh (left) and Target Mesh (right)](/home/ab/assignment2/vis_fit_data/combined_meshes.gif)


Reconstructing 3D from a single view.
===============================================================================

Image to voxel grid.
-------------------------------------------------------------------------------

<head>
    <style>
        .image-container .image img {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }
    </style>
</head>
<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_0.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_vox/GroundTruth_0.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_vox/Predicted_0.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>

<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_27.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_vox/GroundTruth_27.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_vox/Predicted_27.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>

<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_9.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_vox/GroundTruth_9.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_vox/Predicted_9.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>


Image to point cloud.
-------------------------------------------------------------------------------

<head>
    <style>
        .image-container .image img {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }
    </style>
</head>
<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_40.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/GroundTruth_40.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/CMSC848F/assignment2/vis_point/40_predicted_pointcloud.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>

<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_240.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/GroundTruth_240.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/CMSC848F/assignment2/vis_point/240_predicted_pointcloud.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>

<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_120.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/GroundTruth_120.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/CMSC848F/assignment2/vis_point/120_predicted_pointcloud.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>



Image to mesh.
-------------------------------------------------------------------------------

<head>
    <style>
        .image-container .image img {
            width: 300px;
            height: 300px;
            object-fit: cover;
        }
    </style>
</head>
<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_0.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_mesh/GroundTruth_0.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_mesh/Predicted_0.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>

<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_30.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_mesh/GroundTruth_30.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_mesh/Predicted_30.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>

<div class="image-container">
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_point/RGB_240.png" alt="Input RGB Image">
        <p>Input RGB Image</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_mesh/GroundTruth_240.gif" alt="Ground Truth">
        <p>Ground Truth</p>
    </div>
    <div class="image">
        <img src="/home/ab/assignment2/Visualization/Visual_mesh/Predicted_240.gif" alt="Predicted">
        <p>Predicted</p>
    </div>
</div>


Quantitative comparisons.
-------------------------------------------------------------------------------

![Voxel Evaluation](/home/ab/assignment2/Visualization/png2jpg/eval_vox.jpg)
![Point Cloud Evaluation](/home/ab/assignment2/Visualization/png2jpg/eval_point.jpg)
![Mesh Evaluation](/home/ab/assignment2/Visualization/png2jpg/eval_mesh.jpg)


Voxel Reconstruction: The F1 score starts just above 20 and steadily increases, reaching slightly above 70 by a threshold of 0.050.

Point Cloud Reconstruction: The F1 score starts just above 10 and exhibits a rapid growth, reaching close to 90 by a threshold of 0.050. This indicates a more consistent performance across various thresholds.

Mesh Reconstruction: The F1 score starts just above 20, similar to voxel reconstruction, and increases in a steady manner, approaching 80 by a threshold of 0.050.

The point cloud reconstruction demonstrates the highest F1 scores across the range of thresholds, indicating it to be the most accurate method among the three in terms of precision and recall. This suggests that point cloud reconstruction might be more adept at capturing detailed spatial information from the input data.

The mesh reconstruction is after the point cloud performance, showing its capability to approximate 3D structures accurately, especially at higher thresholds. But the results are not accurate in detailing the construction.

Voxel reconstruction, while still effective in some cases, lags behind the other two methods in terms of F1 scores. Also, in some reconstructions some part of the object is completely missing. This might be due to the inherent nature of voxel-based methods where the 3D space is discretized, possibly leading to some loss of detail. Also, I could not evaluate all the 678 input images to voxel grids that might have been the problem.
    
In summary, while all three methods show increasing accuracy as the threshold rises, the point cloud reconstruction stands out as the most effective, followed by mesh reconstruction. Voxel-based methods, though useful, may not capture as much detail as the other two. 

Analyse effects of hyperparameters variations.
-------------------------------------------------------------------------------
I was not able to change these hyperparameters multiple times and see the effects on predictions because of the setup problem but here I am giving review of how that would have changed the predictions.

**Number of points** : This determines the number of points used in a point cloud representation. This hyperparameter could be crucial for capturing the detail and granularity of the 3D structures in point cloud-based methods. With the increase in n_points, the accuracy of the model might improve due to the richer representation. However, the computational cost, both in terms of time and memory, might increase significantly.Higher n_points could capture finer details, but it could also introduce noise or irregularities in the reconstruction.

**Voxel Size** : voxel_size determines the resolution of the voxel grid used in 3D structures. As voxel_size decreases (finer resolution), there may be an improvement in the model's accuracy or reconstruction fidelity. However, smaller voxels can lead to a larger voxel grid, increasing computational demands and memory usage. Smaller voxels can capture intricate details, but if too small relative to the input data's resolution, they may introduce ambiguity or noise in the model's representation.

**w_chamfer** : For low values the model might underemphasize vertex accuracy, leading to potential mismatches between the generated and target meshes. While the overall shape might be retained, finer details could be lost. For high w_chamfer values The model will place increased importance on matching individual vertices. This might lead to highly accurate vertex placements but could also result in the model getting stuck in local minima or overfitting to training data.

**Initial Mesh** : I could not try the training with different meshes but when reconstructing or generating 3D meshes, the initial mesh provides a starting point for the optimization process. With fewer vertices and faces, the mesh has limited expressiveness. The advantage might be faster computation, but the model might struggle to capture intricate details of complex shapes. The resultant meshes might be more abstract or overly smoothed. I could not try the training with different meshes but

Interpret you model.
-------------------------------------------------------------------------------

Here are some Visualization ideas that might be useful to interpret the model:

https://arxiv.org/pdf/1612.00593.pdf

This paper has used different methods to visualize and interpret their results as follows:

Semantic Segmentation Results:
Input Point Cloud: The top row showcases the original point cloud with color information, which represents the raw data input to the model.
Output Semantic Segmentation: The bottom row displays the result of the semantic segmentation on points, visualized from the same camera viewpoint as the input. This visualization emphasizes the comparison between the raw data and the model's understanding of each point's semantic label.

PointNet Robustness Test:
Delete Points: This test removes points from the original dataset. "Furthest" implies that the remaining points are chosen based on maximum distances between them to cover the shape as much as possible.
Outliers: This test inserts outlier points that are scattered uniformly within a unit sphere, challenging the model's ability to distinguish noise from actual shape data.
Gaussian Noise: This test adds Gaussian noise to each point individually. It evaluates how resilient the model is to minor disruptions in the data.

Critical Points and Upper Bound Shape:
Critical Points: Essential points that are needed to determine the global shape feature of a given shape.
Upper Bound Shape: Defines the boundary of shape variations that the model would still recognize as the same global shape feature.
Intermediate Point Clouds: Any point cloud configuration between the critical points and the upper bound shape is interpreted by the model as having the same global shape feature.
Color-coded Depth Information: Provides a sense of depth in the visualization by varying colors based on the distance or depth of the points in the 3D space.

Shape correspondence : The primary goal of this visualization is to show how specific points or features on one object correspond to points or features on another object.
I had done something similar for one my projects earlier. 
feature extraction and matching can be done for better visualization. We can plot epilines to see how and which features are matching between ground truth and predicted image.


![Epilines](/home/ab/assignment2/Visualization/png2jpg_1/epilines.jpg)
![Feature Matching](/home/ab/assignment2/Visualization/png2jpg_1/Matched_features.jpg)


<style>
    .image-container {
        display: flex; 
        justify-content: space-between; 
    }

    .image {
        text-align: center;
    }
</style>


<!-- Markdeep: --><style class="fallback">body{visibility:hidden;white-space:pre;font-family:monospace}</style><script src="markdeep.min.js" charset="utf-8"></script><script src="https://morgan3d.github.io/markdeep/latest/markdeep.min.js?" charset="utf-8"></script><script>window.alreadyProcessedMarkdeep||(document.body.style.visibility="visible")</script>

