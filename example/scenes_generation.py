from visualizer import create_viewer
from params_parser import ParamParser
import numpy as np
from pathlib import Path
import os
import hppfcl
from panda_wrapper import load_reduced_panda
import pinocchio as pin
import random
from scipy.spatial.distance import cdist
import json
from PIL import Image

# Configuration
n_scenes = 1000
output_dir = Path("generated_scenes")
output_dir.mkdir(exist_ok=True)

# Shape definitions
shape_definitions = [("Box", hppfcl.Box(0.2, 0.2, 0.2), 0.4)]
workspace_min = np.array([0.3, -0.8, 0.0])
workspace_max = np.array([0.3, 0.8, 0.8])
colors = [
    [0.2, 0.2, 0.8, 0.7], [0.8, 0.2, 0.2, 0.7],
    [0.2, 0.8, 0.2, 0.7], [0.8, 0.8, 0.2, 0.7],
    [0.2, 0.8, 0.8, 0.7]
]

scene_data = {}

for scene_idx in range(n_scenes):
    # Fresh models for each scene
    rmodel, cmodel, vmodel = load_reduced_panda()
    
    scene_name = f"scene_{scene_idx:04d}"
    
    # Obstacle generation
    num_obstacles = random.randint(1, 5)
    obstacles = []
    
    for obs_idx in range(num_obstacles):
        shape_name, obs_hppfcl, min_dist = random.choice(shape_definitions)
        
        while True:
            random_pos = np.array([
                random.uniform(workspace_min[0], workspace_max[0]),
                random.uniform(workspace_min[1], workspace_max[1]),
                random.uniform(workspace_min[2], workspace_max[2])
            ])
            
            if not obstacles or all(
                np.linalg.norm(random_pos - obs[1]) >= min_dist 
                for obs in obstacles
            ):
                break
        
        color = colors[obs_idx % len(colors)]
        obstacles.append((shape_name, random_pos, obs_hppfcl, color))
        
        # Add to model
        Mobs = pin.SE3(np.eye(3), random_pos)
        obs_id_frame = rmodel.addFrame(
            pin.Frame(f"obs_{obs_idx}", 0, 0, Mobs, pin.OP_FRAME)
        )
        obs_geom = pin.GeometryObject(
            f"obs_{obs_idx}_{shape_name}", 0, obs_id_frame,
            rmodel.frames[obs_id_frame].placement, obs_hppfcl
        )
        obs_geom.meshColor = np.array(color)
        cmodel.addGeometryObject(obs_geom)
    
    # Display and save
    q = pin.neutral(rmodel)
    pin.framesForwardKinematics(rmodel, rmodel.createData(), q)  # Proper kinematics init
    vis = create_viewer(rmodel, cmodel, vmodel)
    vis.display(q)
    img = vis.viewer.get_image()
    img.save(f"generated_scenes/{scene_name}.png")
    # input()
    
    # Store data
    scene_data[scene_name] = {
        "image_path": f"generated_scenes/{scene_name}.png",
        "obstacles": [{
            "type": shape_name,
            "position": pos.tolist(),
            "color": color,
            "shape_params": {"width": 0.2, "depth": 0.2, "height": 0.2}
        } for (shape_name, pos, _, color) in obstacles]
    }

    # Explicit cleanup
    del vis
    del rmodel, cmodel, vmodel

# Save metadata
with open(output_dir / "scenes_data.json", "w") as f:
    json.dump(scene_data, f, indent=2)

print(f"Successfully generated {n_scenes} scenes")