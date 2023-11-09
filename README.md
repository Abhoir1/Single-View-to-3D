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

Look for "combined_voxel.gif" in the root directory, which visualizes the voxel fitting process.

#### Point Representation

```
python fit_data.py --type 'point'
```

"combined_point.gif" will be generated in the root directory, showing the point fitting

#### Mesh Representation

```
python fit_data.py --type 'mesh'
```

"combined_mesh.gif" will be generated in the root directory, showing the point fitting

<img src ="Picture5.jpg" width=400/>
