#!/home/logesh/mujoco_ws/mujoco_env/bin/python3

"""Implementation of IBVS on Mujoco with Kuka iiwa robotic arm - aruco target"""

import cv2
import mujoco
import numpy as np
import mujoco.viewer


class IBVS_MJ:
    def __init__(self):
        
        # mujoco model declaration
        self.model = mujoco.MjModel.from_xml_path("./kuka_iiwa_14/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        # model parameters | ids
        self.q_home = self.model.key("home").qpos
        self.rgbCamId =self.model.camera("camera_rgb").id
        
        # fixed camera renderer | parameters
        self.frame_height = 480
        self.frame_width = 640
        self.renderer = mujoco.Renderer(self.model, height=self.frame_height, width=self.frame_width)
        self.show_renderedImg = True
        self.latestImg = None
        fovy = self.model.camera("camera_rgb").fovy[0]
        fovy_rad = np.deg2rad(fovy)
        f = 0.5 * self.frame_height / np.tan(fovy_rad / 2)
        self.K = np.array([
            [f, 0, (self.frame_width/2)],
            [0, f, (self.frame_height/2)],
            [0, 0, 1]
        ])
        self.camDist = np.zeros((1, 5))

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
        self.desiredCorners = np.array([
            [],
            [],
            [],
            []
        ])
        self.curCorners = None
        self.arucoPose = None
        self.aruco_corner_depth = np.array([-1.0, -1.0, -1.0, -1.0])
    
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