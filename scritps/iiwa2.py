#!/home/logesh/mujoco_ws/mujoco_env/bin/python3

"""This is an example script for testing different approches in mujoco"""

import cv2
import mujoco
import numpy as np
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("./kuka_iiwa_14/scene.xml")
data = mujoco.MjData(model)

# print(model)
# print(data)

# for idx in range(model.nu):
#     print(f"idx_{idx} - {model.actuator(idx).name}")


# for i in range(model.nu):
#     print(model.actuator(i).name, model.actuator(i).trntype)

# for i in range(model.nbody):
#     print(i, '->',model.body(i).name)

# body names
# body_names = [
#     "base",
#     "link1",
#     "link2",
#     "link3",
#     "link4",
#     "link5",
#     "link6",
#     "link7",
# ]
# # get body ids
# body_ids = [model.body(name).id for name in body_names]
# # gravity compensation
# model.body_gravcomp[body_ids] = 1.0


renderer = mujoco.Renderer(model, height=480, width=640)
rs_rgbCam = model.camera("camera_rgb").id

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    # Reset the free camera.
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    q_start = data.qpos.copy()
    q_home = model.key("home").qpos
    tar_vel = np.array([0.1, 0, 0, 0, 0, 0, 0])

    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_CAMERA

    while viewer.is_running():

        data.ctrl[:] = q_home

        renderer.update_scene(data, camera=rs_rgbCam)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("", bgr)
        cv2.waitKey(1)

        # q = data.qpos.copy()
        # mujoco.mj_integratePos(model, q, tar_vel, 1)

        mujoco.mj_step(model, data)
        viewer.sync()
