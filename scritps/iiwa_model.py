#!/home/logesh/mujoco_ws/mujoco_env/bin/python3

import cv2
import mujoco
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation

gravity_compensation = True
integration_dt = 0.1
first_itr = True

# aruco pose detection
def find_aruco_pose(frame, camera_matrix, dist_coeffs, marker_length, marker_dict=cv2.aruco.DICT_5X5_1000):
    """Finds aruco marker pose."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)          
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    (corners, ids, rejected) = detector.detectMarkers(gray)

    # pose of aruco
    marker_size = marker_length  # in mm
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    R_target2cam, t_target2cam = None, None
    if np.all(ids is not None):
        for i in range(len(ids)):
            ret, rvec, tvec = cv2.solvePnP(marker_points, corners[i], camera_matrix, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)    
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec,30,3)
            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec.squeeze()

    return R_target2cam, t_target2cam

def main():
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)
    rs_rgbCam = model.camera("camera_rgb").id

    fovy = model.camera("camera_rgb").fovy[0]
    fovy_rad = np.deg2rad(fovy)
    f = 0.5 * 480 / np.tan(fovy_rad / 2)
    K = np.array([[f, 0, (640/2)], [0, f, (480/2)], [0, 0, 1]])
    dist = np.zeros((1, 5))

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

    home_pos = model.key("home").qpos
    ee_id = model.body('link7').id
    base_id = model.body('link1').id
    cam_id = model.camera('camera_rgb').id
    aruco_id = model.geom('aruco').id

    error = np.zeros((6))
    error_pos = error[0:3]
    error_ori = error[3:6]
    site_quat = np.zeros((4))
    site_quat_conj = np.zeros((4))
    error_quat = np.zeros(4)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            global first_itr

            # --- GRAVITY COMPENSATION ---
            data.qvel[:] = 0
            data.qacc[:] = 0
            # data.qpos = home_pos
            
            if (first_itr):
                first_itr = False
                mujoco.mj_rne(model, data, 0, data.qfrc_inverse)
                tau_g = data.qfrc_inverse[dof_ids]
                # apply torques
                data.ctrl[actuator_ids] = tau_g
                mujoco.mj_step(model, data)
                viewer.sync()
                continue

            # eye camera rendering
            renderer.update_scene(data, camera=rs_rgbCam)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # find aruco marker pose
            arucoPose = find_aruco_pose(bgr, K, dist, 0.050)
            cTo = np.eye(4)                                 # camera to object transpose
            cTo[0:3, 0:3] = arucoPose[0]
            cTo[0:3, 3] = arucoPose[1]
            # show image
            cv2.imshow("rgb camera", bgr)
            cv2.waitKey(1)

            
            ## visual servoing

            curEE_trans = data.xpos[ee_id]
            curEE_rotm = Rotation.from_quat(data.xquat[ee_id]).as_matrix()
            # bTe (base to ee transform)
            bTe = np.eye(4)
            bTe[0:3, 0:3] = curEE_rotm
            bTe[0:3, 3] = curEE_trans

            # camera transform (from base or world)
            bTc = np.eye(4)
            curCam_t = data.cam_xpos[cam_id]
            curCam_r = np.reshape(data.cam_xmat[cam_id], (3, 3))
            bTc[0:3, 0:3] = curCam_r
            bTc[0:3, 3] = curCam_t
    
            # base to object transform
            bTo = bTc @ cTo         # TODO: cTo is slighty wrong
            bTo[2, 3] = 0.5

            # hard coding target position
            # bTo[0:3, 3] = data.geom_xpos[aruco_id]
            # bTo[0:3, 0:3] = np.reshape(data.geom_xmat[aruco_id], (3,3))
            # bTo[2, 3] = 0.5
            
            # Position error.
            error_pos[:] = bTo[0:3, 3] - bTc[0:3, 3]
            print(error_pos)

            # Orientation error.
            # mujoco.mju_mat2Quat(site_quat, data.cam_xmat[cam_id])
            # mujoco.mju_negQuat(site_quat_conj, site_quat)
            # mujoco.mju_mulQuat(error_quat, data.xquat[ee_id], site_quat_conj)
            # mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, robotJacob[:3], robotJacob[3:], site_id)
            dq = robotJacob.T @ np.linalg.solve(robotJacob @ robotJacob.T, error)
            # print(dq)
            
            # apply velocity
            data.qvel = dq * 5.0
            
            ##


            # joint vel + ee vel control
            """
            targetVel = np.zeros((6))
            targetVel[0] = 0.1
            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, robotJacob[:3], robotJacob[3:], site_id)
            dq = np.linalg.pinv(robotJacob) @ targetVel
            data.qvel = dq
            """

            # target velocity generation using qpos error
            """
            position_error = np.subtract(tar, data.qpos)
            kp = 5.0
            targetVel = kp * position_error
            print(np.round(position_error, 3))
            data.qvel = targetVel
            """

            # Integrate joint velocities to obtain joint positions.     # this did not work
            # q = data.qpos.copy()
            # mujoco.mj_integratePos(model, q, dq, integration_dt)
            # data.qpos = np.zeros(7)

            # position control
            """
            tarPosi
            """
            
            mujoco.mj_rne(model, data, 0, data.qfrc_inverse)

            tau_g = data.qfrc_inverse[dof_ids]

            # apply torques
            data.ctrl[actuator_ids] = tau_g

            mujoco.mj_step(model, data)
            viewer.sync()
    

if __name__ == "__main__":
    main()