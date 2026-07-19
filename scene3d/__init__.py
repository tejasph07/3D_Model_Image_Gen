"""scene3d: text-to-3D-scene generator producing OpenPose skeletons.

Turn dead-simple 3-token scene scripts into physically-grounded 3D scenes of
humanoids and common objects, rendered as OpenPose BODY_25 (+hands) skeleton
images and JSON for conditioning image models.
"""
from .parser import parse, ParseError
from .solver import Scene, SceneError, build_scene
from .render import render
from .camera import make_camera
from .openpose import scene_json, write_json

__all__ = ["parse", "ParseError", "Scene", "SceneError", "build_scene",
           "render", "make_camera", "scene_json", "write_json"]
__version__ = "0.1.0"
