import numpy as np
from pytransform3d.rotations import matrix_from_euler
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller
from contact_lfd.LfDusingEC.utils.utils_robotMath import RtToTrans

# from trac_ik_python.trac_ik import IK
# ik_solver = IK("panda_link0",'panda_hand_tcp')


arm_ctrl = Arm_Controller(gripper=True)

num_poses = 10
angles = np.linspace(0, 2*np.pi, num_poses)

height = 0.25
radius = 0.12
ee_ht_ee_traj = []

center_point = np.array([0.434576, 0.0720451, height])
init_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
init_base_ht_ee = RtToTrans(R=init_R, t=center_point)
arm_ctrl.move_to_carte_position(init_base_ht_ee, completion_time=5)

for theta in angles:
    x = radius * np.cos(theta+np.pi/2)
    y = radius * np.sin(theta+np.pi/2)

    if theta < np.pi:
        yaw = theta - np.pi/2
        pitch = np.arctan2(radius, height)
    else:
        yaw = theta - 3*np.pi/2
        pitch = -1*np.arctan2(radius, height)

    diff_R = matrix_from_euler([yaw, pitch, 0], 2, 1, 0, False)

    ee_ht_ee = RtToTrans(R=diff_R, t=[x, y, 0])
    ee_ht_ee_traj.append(ee_ht_ee)

    d_base_ht_ee = init_base_ht_ee @ ee_ht_ee
    arm_ctrl.move_to_carte_position(d_base_ht_ee, completion_time=5)

print("test")


