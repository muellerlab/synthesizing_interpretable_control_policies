"""On every iteration, improve robot_v1 over the robot_vX from previous iterations.
Make only small changes. Try to make the code short. 

Each function should return a list of integers. The integers should be between 0 and 19 (inclusive). Duplicates are allowed. Return only one completed function with no additional text, explanation, or formatting.

An example of a good design is: [0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 18, 9, 8, 9, 9, 8]
An example of a bad design is: [0, 1, 2, 3]
"""
import funsearch
import re
from typing import List
import requests

# Ignore these 3 variables
METHOD_MATCHER = re.compile(r"def robot_v\d\(.*?\) -> List\[int\]:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
METHOD_NAME_MATCHER = re.compile(r"robot_v\d+")
method_str = "def robot_v"

@funsearch.run
def evaluate_robot(task="RidgedTerrainTask") -> float:
  """Returns the best reward managed by the robot design. Done via calling the robogrammar API
  """
  # Type checking
  robo_design = robot()
  
  if not isinstance(robo_design, list) or not all([isinstance(x, int) for x in robo_design]):
      return 0
  
  url = "http://127.0.0.1:5555/simulate"
  payload = {
      "task": task,
      "grammar_file": "data/designs/grammar_apr30.dot",
      "rule_sequence": robo_design,
      "jobs": 8,
      "optim": True,
      "episodes": 1, # using multiple episodes causes FCValueEstimator to crash apparently --- keep at 1
      "episode_len": 30
   }
  response = requests.post(url, json=payload)
  data = response.json()
  optimization_result = data["distance_travelled"]
  return optimization_result

@funsearch.evolve
def robot() -> List[int]:
  """Returns a list of numbers between 1 and 19 (inclusive) that represent the robot design.
  """
  design = [0]
  
  return design