#!/home/logesh/mujoco_ws/mujoco_env/bin/python3

import mujoco
import numpy as np
import mujoco.viewer

def main():
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:

        while viewer.is_running():
            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()

    

if __name__ == "__main__":
    main()