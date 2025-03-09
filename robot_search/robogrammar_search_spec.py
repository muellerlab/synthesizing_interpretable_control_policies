"""Finds a policy for designing a robot for the given task.

On every iteration, improve robot_v1 over the robot_vX from previous iterations.
Make only small changes. Try to make the code short. 

The robot is formed through a list of numbers, and are automatically designed symmetrically.
"""

import numpy as np
import funsearch
import re
from typing import List
import requests


METHOD_MATCHER = re.compile(r"def robot_v\d\(.*?\) -> float:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
METHOD_NAME_MATCHER = re.compile(r"robot_v\d+")
method_str = "def robot_v"

@funsearch.run
def evaluate_robot(design) -> float:
  """Returns the best reward managed by the robot design. Done via calling the robogrammar API
  """
  url = "http://127.0.0.1:5555/simulate"
  payload = {
      "task": "RidgedTerrainTask",
      "grammar_file": "data/designs/grammar_apr30.dot",
      "rule_sequence": design,
      "jobs": 8,
      "optim": True,
      "episodes": 1, # using multiple episodes causes FCValueEstimator to crash apparently --- keep at 1
      "episode_len": 128
   }
  response = requests.post(url, json=payload)
  data = response.json()
  optimization_result = data["optimization_result"]  
  return optimization_result

@funsearch.evolve
def robot() -> List[int]:
  """Returns a list of numbers corresponding to a robot
  """
  design = [0]
  
  return design