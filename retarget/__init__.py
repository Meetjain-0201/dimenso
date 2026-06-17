"""Retarget layer: human hand landmarks -> G1 upper-body joint targets.

P2. Two stages:
  * kinematics: human wrist pose + grasp -> G1 end-effector target + grasp state
  * ik_solver:  G1 EE target -> G1 upper-body joint positions (Pink IK)
"""
