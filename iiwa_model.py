#!/home/logesh/mujoco_ws/mujoco_env/bin/python3

import cv2
import mujoco
import numpy as np
import mujoco.viewer

gravity_compensation = True
integration_dt = 0.1

def main():
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)
    rs_rgbCam = model.camera("camera_rgb").id

    # model.opt.gravity = (0,0,0)
    # print(model.opt.gravity)

    key_id = model.key("home").id
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7"
    ]
    
    site_id = model.site("attachment_site").id
    robotJacob = np.zeros((6, model.nv))    # (6, 7) in this case

    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE


        while viewer.is_running():
            # --- GRAVITY COMPENSATION ---
            data.qvel[:] = 0
            data.qacc[:] = 0
            
            # joint vel + ee vel control
            targetVel = np.zeros((6))
            targetVel[0] = 0.1
            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, robotJacob[:3], robotJacob[3:], site_id)
            dq = np.linalg.pinv(robotJacob) @ targetVel
            data.qvel = dq

            # Integrate joint velocities to obtain joint positions.     # this did not work
            # q = data.qpos.copy()
            # mujoco.mj_integratePos(model, q, dq, integration_dt)
            # data.qpos = np.zeros(7)

            # position control
            """
            tarPosi
            """

            renderer.update_scene(data, camera=rs_rgbCam)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("rgb camera", bgr)
            cv2.waitKey(1)
            
            mujoco.mj_rne(model, data, 0, data.qfrc_inverse)

            tau_g = data.qfrc_inverse[dof_ids]

            # apply torques
            data.ctrl[actuator_ids] = tau_g

            mujoco.mj_step(model, data)
            viewer.sync()

    

if __name__ == "__main__":
    main()