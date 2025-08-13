import time
import zerorpc
import numpy as np
from threading import Thread
from FlexivPy.controllers.mpc import UnconstrainedReachingMPC
from FlexivPy.robot.interface import FlexivDDSClient
from FlexivPy.robot.interface import FlexivCmd
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rot

class ZeroRPCWrapper:
    def __init__(self, controller):
        self.robot_interface = FlexivDDSClient()
        self.mpc = controller 
        self.running = False
        self.control_thread = Thread(target=self.control_loop)

    def start(self):
        if self.running:
            print("Controller is already running.")
            return
        self.running = True
        self.mpc.initialize(q0=np.array(self.robot_interface.get_robot_state().q))
        self.mpc.setHomeCommand()
        self.control_thread.start()

    def stop(self):
        if not self.running:
            print("Controller is not running.")
            return
        self.running = False
        self.control_thread.join()
        cmd = FlexivCmd()
        self.robot_interface.set_cmd(cmd)
    
    def control_loop(self):
        while self.running:
            tic = time.time()
            state = self.robot_interface.get_robot_state()
            q, dq = np.array(state.q), np.array(state.dq)
            cmd = self.mpc.updateAndSolve(q, dq)
            command = FlexivCmd(
                q=cmd['q'].tolist(),
                dq=cmd['dq'].tolist(),
                mode=3,
            )
            self.robot_interface.set_cmd(command)
            while time.time() - tic < self.mpc.dt:
                time.sleep(0.0001)

    def goHome(self):
        self.mpc.setHomeCommand()

    def setAbsoluteAction(self, action):
        action = np.array(action)
        assert action.shape == (4,4), "Action must be a 4x4 matrix"
        self.mpc.setSinglePoseAction(action)

    def setAbsoluteActions(self, actions):
        assert len(actions) == self.mpc.HORIZON+1, "Actions must match MPC horizon"
        actions = [np.array(action) for action in actions]
        self.mpc.setDesiredPoses(actions)

    def setRelativeAction(self, action):
        assert len(action) == 6, "Action must be a 6D vector"
        action = np.array(action)
        delta_trans = action[:3]
        delta_rot = action[3:]
        T_desired = self.mpc.getDesiredPoses()[-1]
        R_target = Rot.from_euler('xyz', delta_rot).as_matrix()@T_desired[:3, :3]
        t_target = T_desired[:3,-1] + delta_trans
        T_desired[:3, :3] = R_target
        T_desired[:3, -1] = t_target
        self.mpc.setSinglePoseAction(T_desired)

    def getState(self):
        state = self.robot_interface.get_robot_state()
        pin.forwardKinematics(self.mpc.rmodel, self.mpc.rdata, np.array(state.q))
        pin.updateFramePlacements(self.mpc.rmodel, self.mpc.rdata)
        efPos0 = self.mpc.rdata.oMf[self.mpc.efId].translation
        efRot0 = self.mpc.rdata.oMf[self.mpc.efId].rotation
        return {
            'q': state.q,
            'dq': state.dq,
            'tau': state.tau,
            'ef_translation': efPos0.tolist(),
            'ef_rotation': efRot0.tolist(),
            'ft_sensor': state.ft_sensor,
            'g_force': state.g_force,
            'g_width': state.g_width,
            'g_state': state.g_state,
            'g_moving': state.g_moving,
        }
    
    def getMPCInfo(self):
        return {
            'HORIZON': self.mpc.HORIZON,
            'dt': self.mpc.dt,
            'current_action': self.mpc.getDesiredPoses()[-1].tolist(),
        }

def main():
    mpc = UnconstrainedReachingMPC()
    controller = ZeroRPCWrapper(mpc)
    controller.start()
    zerorpc_server = zerorpc.Server(controller)
    zerorpc_server.bind("tcp://0.0.0.0:4242")
    zerorpc_server.run()

if __name__ == '__main__':
    main()