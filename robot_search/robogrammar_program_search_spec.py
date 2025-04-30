"""
On every iteration, improve robot_v1 over the previous robot_vX from previous iterations.

Make only small changes. Try to make the code short.

The robot function takes in a sequence of integers and returns a new integer from 0 to 20.
The function should return -1 in order to indicate that you want to end the sequence. 

You may use duplicate integers and can use any method you like to come up with the new integer.
"""

import funsearch
import re
from typing import List
import requests
import math
import numpy as np

# Regex Patterns --- Ignore these 3 variables
METHOD_MATCHER = re.compile(r"def robot_v\d\(input_list: List\[int\]\) -> int:(?:\s*(?:[ \t]*(?!def|#|`|').*(?:\n|$)))+")
METHOD_NAME_MATCHER = re.compile(r"robot_v\d+")
method_str = "def robot_v"

@funsearch.run
def evaluate_robot(task="RidgedTerrainTask") -> float:
  """Returns the best reward managed by the robot design. Done via calling the robogrammar API
  """
  rule_seq = [0] # Starting with 0
  filter_after_each_entry = False
  rule_gen_max_depth = 25
  
  i = 0
  while i < rule_gen_max_depth and rule_seq[-1] != -1:
    rule_seq.append(robot(rule_seq[:]))
    if filter_after_each_entry and rule_seq[-1] != -1:
        rule_seq = filter(rule_seq)
    i += 1 
  
  def filter(input_list: List[int]) -> List[int]:
      return [k for k in input_list if k in range(0, 20)]
  
  final_rule_seq = filter(rule_seq) # Filter the rule sequence
  
  url = "http://127.0.0.1:5555/simulate"
  payload = {
      "task": task,
      "grammar_file": "data/designs/grammar_apr30.dot",
      "rule_sequence": final_rule_seq,
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
def robot(input_list: List[int]) -> int:
  """Returns the next number that should be appended to this sequence. Return -1 to stop the sequence.
  Feel free to use any logic or functions you like to come up with the next number.
  """
  return 0