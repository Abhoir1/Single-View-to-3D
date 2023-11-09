# Single View to 3D README

## Configuration

Update the `data_location.py` file with the root location of your dataset to match your system's directory structure.

## Execution Instructions

Execute the following commands in your terminal. If necessary, replace `python` with the version specific to your system (e.g., `python3`).

### Exploring Loss Functions

#### Voxel Representation
```
python fit_data.py --type 'vox'
```
<img src ="vis_fit_data/combined_voxel.gif" height=300/>

#### Point Representation

```
python fit_data.py --type 'point'
```
<img src ="vis_fit_data/combined_pointclouds.gif" height=300/>
#### Mesh Representation

```
python fit_data.py --type 'mesh'
```
<img src ="vis_fit_data/combined_meshes.gif" height=300/>



