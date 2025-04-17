import example_robot_data as robex
import numpy as np
import pinocchio as pin


def load_panda():
    panda = robex.load("panda_collision")
    rmodel, cmodel, vmodel = panda.model, panda.collision_model, panda.visual_model
    return rmodel, cmodel, vmodel

def load_reduced_panda():
    rmodel, cmodel, vmodel = load_panda()
    geom_models = [vmodel, cmodel]
    rmodel, geometric_models_reduced = pin.buildReducedModel(
        rmodel,
        list_of_geom_models=geom_models,
        list_of_joints_to_lock=[7, 8],
        reference_configuration=np.array([-0.6513877410293797, 1.3677075286603906, -0.17736737718858037, -0.3973375018143172, -0.11554961778792178, 1.2408486160482337, 8.644879755868687e-05, 0.0, 0.0])
    )
    
    vmodel, cmodel = geometric_models_reduced[0], geometric_models_reduced[1]
    return rmodel, cmodel, vmodel