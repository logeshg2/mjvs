#!/home/logesh/mujoco_ws/mujoco_env/bin/python3

"""Implementation of IBVS on Mujoco with Kuka iiwa robotic arm - aruco target"""

import cv2
import mujoco
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation


class IBVS_MJ:
    def __init__(self):
        
        # mujoco model declaration
        self.model = mujoco.MjModel.from_xml_path("./kuka_iiwa_14/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        # model parameters | ids
        self.ee_id = self.model.body('link7').id
        self.q_home = self.model.key("home").qpos
        self.rgbCamId =self.model.camera("camera_rgb").id
        
        # default starting state
        self.dt = 1.0
        # self.data.ctrl[:] = self.q_home
        mujoco.mj_forward(self.model, self.data)

        # fixed camera renderer | parameters
        self.frame_height = 480
        self.frame_width = 640
        self.renderer = mujoco.Renderer(self.model, height=self.frame_height, width=self.frame_width)
        self.show_renderedImg = True
        self.latestImg = None
        # camera intrinsic's
        fovy = self.model.camera("camera_rgb").fovy[0]
        fovy_rad = np.deg2rad(fovy)
        f = 0.5 * self.frame_height / np.tan(fovy_rad / 2)
        self.K = np.array([
            [f, 0, (self.frame_width/2)],
            [0, f, (self.frame_height/2)],
            [0, 0, 1]
        ])
        self.Kinv = np.linalg.inv(self.K)
        self.camDist = np.zeros((1, 5))
        
        # camera extrinsic's
        # world to ee (link 6) transform 
        wTe = np.eye(4)
        wTe[0:3, 3] = self.data.xpos[self.data.body('link7').id]
        wTe[0:3, 0:3] = np.array(self.data.xmat[self.data.body('link7').id]).reshape((3,3))
        eTw = np.linalg.pinv(wTe)
        # world to camera (camera_rgb) transform
        wTc = np.eye(4)
        wTc[0:3, 3] = self.data.cam_xpos[self.rgbCamId]
        wTc[0:3, 0:3] = np.array(self.data.cam_xmat[self.rgbCamId]).reshape((3,3))
        # ee to camera transform   
        self.eTc = eTw @ wTc

        # robot movement variables
        self.tar_q = np.array([0.0 for i in range(self.model.njnt)])
        self.tar_ee_dot = np.array([0.0 for i in range(6)])
        self.tar_q_dot = np.array([0.0 for i in range(self.model.njnt)])

        # aruco marker parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        aruco_param = cv2.aruco.DetectorParameters()
        self.arucoDetector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)
        self.markerLength = 0.100
        self.object_points = np.array([
            [-self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, -self.markerLength / 2, 0],
            [self.markerLength / 2, self.markerLength / 2, 0],
            [-self.markerLength / 2, self.markerLength / 2, 0]
        ])
        self.tar_top_left = np.array([243, 473])
        self.tar_top_right = np.array([247, 316])
        self.tar_bottom_right = np.array([395, 316])
        self.tar_bottom_left = np.array([400, 473])
        self.tar_Z = 0.38099
        self.curCorners = None
        self.arucoPose = None
        self.aruco_corner_depth = np.array([-1.0, -1.0, -1.0, -1.0])
    
        # visual servoing parameters
        self.curCamVel = None
        self.lambdaVar = 0.3
        self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # adjoint transformation (camera frame velocity to end-effector frame velocity transform)
        self.ADeTc = np.zeros((6, 6))
        self.ADeTc[0:3, 0:3] = self.eTc[0:3, 0:3]
        self.ADeTc[3:6, 3:6] = self.eTc[0:3, 0:3]
        eTc_t = self.eTc[0:3, 3]
        etc_x = np.array([               # skew symmetric matrix of translation (eTc.t)
            [0, (-1 * eTc_t[2]), eTc_t[1]],
            [eTc_t[2], 0, (-1 * eTc_t[0])],
            [(-1 * eTc_t[1]), eTc_t[0], 0]
        ])
        self.ADeTc[3:6, 0:3] = etc_x @ self.eTc[0:3, 0:3]
        # compute desired points interaction matrix
        self.desiredIntMat = None
        self.computeDesiredInteractionMat()


    def renderImg(self):
        """Function to update the scene and render the image (EE mounted camera here)"""
        
        self.renderer.update_scene(self.data, camera=self.rgbCamId)
        self.latestImg = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)       # rgb to bgr (default in opencv)

    def detectAruco(self):
        """Function to detect aruco marker corners + finding aruco pose with respect to camera frame"""
        
        grayImg = cv2.cvtColor(self.latestImg, cv2.COLOR_BGR2GRAY)
        self.curCorners, ids, _ = self.arucoDetector.detectMarkers(grayImg)
        if (ids is not None):
            self.curCorners = np.int64(self.curCorners).reshape((4,2))
            cv2.circle(self.latestImg, self.curCorners[0], 3, (0,255,0), -1)
            cv2.circle(self.latestImg, self.curCorners[1], 3, (0,255,0), -1)
            cv2.circle(self.latestImg, self.curCorners[2], 3, (0,255,0), -1)
            cv2.circle(self.latestImg, self.curCorners[3], 3, (0,255,0), -1)

            # target position
            cv2.circle(self.latestImg, self.tar_top_left, 3, (0,0,255), -1)
            cv2.circle(self.latestImg, self.tar_top_right, 3, (0,0,255), -1)
            cv2.circle(self.latestImg, self.tar_bottom_right, 3, (0,0,255), -1)
            cv2.circle(self.latestImg, self.tar_bottom_left, 3, (0,0,255), -1)

            # aruco pose
            ret, rvec, tvec = cv2.solvePnP(self.object_points, np.float64(self.curCorners), self.K, self.camDist, False, cv2.SOLVEPNP_IPPE_SQUARE)
            if (rvec is not None):
                # compute rotation matrix
                rotm, _ = cv2.Rodrigues(rvec)
                # camera to aruco pose
                cTa = np.eye(4)
                cTa[0:3, 3] = tvec.flatten()
                cTa[0:3, 0:3] = rotm

                # draw the pose frame on aruco
                cv2.drawFrameAxes(self.latestImg, self.K, self.camDist, rvec, tvec, 0.1, 2)
                
                # find depth of each corners from camera frame
                for idx, corner in enumerate(self.object_points):
                    # aruco to corner pose
                    aTp = np.eye(4)
                    aTp[0:3, 3] = np.array(corner).flatten()
                    # camera to corner pose
                    cTp = cTa @ aTp
                    # extract depth of the corner point
                    self.aruco_corner_depth[idx] = cTp[2, 3]
            else:
                # pose was not computed
                self.arucoPose = None
                self.aruco_corner_depth = np.array([-1.0, -1.0, -1.0, -1.0])
        else:
            # no aruco detected
            self.curCorners = None
            self.arucoPose = None
            self.aruco_corner_depth = np.array([-1.0, -1.0, -1.0, -1.0])

    def computeImgPointVel(self, des_point, cur_point):
        """
        function to compute image pixel (i.e., change in image place) velocity between two points.
        from 'Visual Servo Control Part I: Basic Approaches' literature -> error = current - desired 
        """

        desPoint = np.array([des_point[0], des_point[1], 1]).reshape(3, 1)
        curPoint = np.array([cur_point[0], cur_point[1], 1]).reshape(3, 1)
        des_xy = self.Kinv @ desPoint
        cur_xy = self.Kinv @ curPoint
        
        des_xy = des_xy.flatten()[:2]
        cur_xy = cur_xy.flatten()[:2]

        pix_vel = np.subtract(np.array(cur_xy), np.array(des_xy))           # difference of points in image plane
        return pix_vel

    def computeDesiredInteractionMat(self):
        """
        Function to compute desired aruco corner points interaction matrix
        This matrix computed will be used for `Approximation of Interaction Matrix`.
        """

        # desire aruco points (corners) interaction matrix
        p1_jac = self.computeInteractionMatrix(self.tar_top_left[0], self.tar_top_left[1], self.tar_Z)
        p2_jac = self.computeInteractionMatrix(self.tar_top_right[0], self.tar_top_right[1], self.tar_Z)
        p3_jac = self.computeInteractionMatrix(self.tar_bottom_right[0], self.tar_bottom_right[1], self.tar_Z)
        p4_jac = self.computeInteractionMatrix(self.tar_bottom_left[0], self.tar_bottom_left[1], self.tar_Z)
        # points jacobian (for all for points) - 8x6 matrix
        # desired points interaction matrix
        self.desiredIntMat = np.vstack([p1_jac, p2_jac, p3_jac, p4_jac])

    def computeInteractionMatrix(self, u, v, Z):
        """
        Function 'computeInteractionMatrix' is used to compute image jacobian (J) or interaction matrix (L) of the given pixel point (u, v).
        
        Args:
            - u : in pixel
            - v : in pixel
            - Z : in meters (depth of point in camera frame)
        
        """
        # compute image coordinates (x, y) from (u, v)
        # x = (u - self.cx) / self.fx
        # y = (v - self.cy) / self.fy
        point = np.array([[u,v,1]]).T
        xy = self.Kinv @ point
        x = xy[0, 0]
        y = xy[1, 0]
        Z = Z

        # image jacobian template(or formula) - 2x6
        img_jacobian = np.array([[(-1/Z), 0, (x/Z), (x*y), -(1+(x*x)), y], 
                                [0, (-1/Z), (y/Z), (1+(y*y)), (-x*y), -x]])
        
        return img_jacobian

    def computeCamVel(self):
        """Function to compute camera velocity based on pixel diff or pixel velocity"""

        # compute pixel velocity
        top_left_pix_vel = self.computeImgPointVel(self.tar_top_left, self.curCorners[0])
        top_right_pix_vel = self.computeImgPointVel(self.tar_top_right, self.curCorners[1])
        bottom_right_pix_vel = self.computeImgPointVel(self.tar_bottom_right, self.curCorners[2])
        bottom_left_pix_vel = self.computeImgPointVel(self.tar_bottom_left, self.curCorners[3])
        # flatten pixel velocities (here - 8x1 vector)
        self.pixelVel = np.array([top_left_pix_vel.flatten(), top_right_pix_vel.flatten(), bottom_right_pix_vel.flatten(), bottom_left_pix_vel.flatten()])
        self.pixelVel = np.array([self.pixelVel.flatten()]).T     # [[u1_dot], [v1_dot], [u2_dot], [v2_dot], [u3_dot], [v3_dot], [u4_dot], [v4_dot]] - 8x1

        # compute interation matrix (combain all for points interation matrix)
        top_left_jacob = self.computeInteractionMatrix(self.curCorners[0][0], self.curCorners[0][1], self.aruco_corner_depth[0])
        top_right_jacob = self.computeInteractionMatrix(self.curCorners[1][0], self.curCorners[1][1], self.aruco_corner_depth[1])
        bottom_right_jacob = self.computeInteractionMatrix(self.curCorners[2][0], self.curCorners[2][1], self.aruco_corner_depth[2])
        bottom_left_jacob = self.computeInteractionMatrix(self.curCorners[3][0], self.curCorners[3][1], self.aruco_corner_depth[3])
        # points jacobian (for all for points) - 8x6 matrix
        # current points jacobian
        pointsJacob = np.vstack([top_left_jacob, top_right_jacob, bottom_right_jacob, bottom_left_jacob])

        # [IMP]
        # Approximation of Interaction Matrix   (8x6)
        approxIntMat = (pointsJacob + self.desiredIntMat) / 2

        # Adaptive gain (lambda_adapt)
        # self.setAdaptiveGain(0.5, 0.3, 30.0)           # default - [1.666, 0.666, 1.666] 
        # tuning adaptive gain parameter using constant lambda
        self.lambdaVar = 0.3                              # uncomment and tune lambda 0, and inf

        # compute camVel 
        # camVel = -1 * self.lambdaVar * (inv(approxIntMat) @ pixelVel)
        camVel = -1 * self.lambdaVar * (np.linalg.pinv(approxIntMat) @ self.pixelVel)        # (6x1) = (6x8) @ (8x1)
        # NOTE: self.lambdaVar is negative for Eye in Hand, and positive for Eye to Hand
        # NOTE: `camVel.flatten()` -> [Vx, Vy, Vz, Wx, Wy, Wz]
        
        # print(np.round(camVel.flatten(), 4))
        
        return camVel.flatten()

    def computeEEVel(self):
        # check aruco corners detection
        if (self.arucoPose is not None and self.aruco_corner_depth[0] != -1):
            # compute desired camera velocity
            self.curCamVel = self.computeCamVel()
            # self.get_logger().info(f"Computed Cam velocity: {self.curCamVel}")

            # camera velocity to end effector velocity
            # using adjoint transformation (Ad_eTc)
            self.ee_vel = (self.ADeTc @ self.curCamVel).flatten()       # (6,)
        else:
            self.ee_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def integrateVel(self, qpos, qvel):
        # update joint position by integrating velocity
        for i in range(6):
            qpos[i] += (self.dt * qvel[i])
        
        return qpos

    def run_sim(self):
        """Main function that loops the entire servoing operation"""

        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) as viewer:
            # reset
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                # render image
                self.renderImg()
                # detect aruco corners + pose
                self.detectAruco()
                # show processed rendered image
                if (self.show_renderedImg):
                    cv2.imshow("EE Cam", self.latestImg)
                    cv2.waitKey(1)

                # ibvs control
                # compute ee velocity
                self.computeEEVel()
                # log (debug)
                print(np.round(self.ee_vel, 4))

                # end effector velocity to joint velocity
                # ee - jacobian
                J_pos = np.zeros((3, self.model.nv))   # linear velocity Jacobian
                J_rot = np.zeros((3, self.model.nv))   # angular velocity Jacobian
                mujoco.mj_jacBody(self.model, self.data, J_pos, J_rot, self.ee_id)
                jac = np.vstack([J_pos, J_rot])
                q_dot = np.linalg.pinv(jac) @ np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
                # compute targe position
                q_tar = self.integrateVel(self.data.qpos.copy(), q_dot)

                # all control (target position)
                self.data.ctrl[:] = self.q_home

                # step sim
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    
    def __del__(self):
        cv2.destroyAllWindows()
        print("Shutting down mujoco sim")


def main():
    obj = IBVS_MJ()
    obj.run_sim()

if __name__ == "__main__":
    main()