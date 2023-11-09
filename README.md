CMSC848F

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

After each evalution set the threshold vs F1 score plot is saved in Visualization/Evaluation directory.
