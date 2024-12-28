from scipy import interpolate

import numpy as np
import pinocchio as pin
from FlexivPy.robot.dds.flexiv_messages import (
    FlexivCmd,
)
from FlexivPy.controllers.utils import *
from pinocchio.visualize import MeshcatVisualizer
import os
from FlexivPy import ASSETS_PATH
from FlexivPy.planners.rrt import RRT

import time


class Get_T_from_controller_no_drift:
    def __init__(self, joy, T0=None, max_v=np.inf):
        self.joy = joy
        self.T0 = T0
        self.max_v = max_v
        # Initialize the desired pose
        self.x0 = 0
        self.y0 = 0.0
        self.z0 = 0.0
        self.R0 = np.eye(3)

    def __call__(self, state):

        if self.T0 is None:
            raise Exception("Initial pose is not provided!")

        rate = 0.25 / 100.0
        
        joy_state = self.joy.getStates()
        left_joy = joy_state["left_joy"]
        right_joy = joy_state["right_joy"]

        if joy_state["right_bumper"] == 0:
            vx_cmd = -1 * right_joy[1]
            vy_cmd = -1 * right_joy[0]
            vz_cmd = 1 * left_joy[0]

            if vx_cmd > self.max_v:
                vx_cmd = self.max_v
            if vx_cmd < -self.max_v:
                vx_cmd = -self.max_v
            if vy_cmd > self.max_v:
                vy_cmd = self.max_v
            if vy_cmd < -self.max_v:
                vy_cmd = -self.max_v

            if np.abs(vx_cmd) < 0.1:
                vx_cmd = 0
            if np.abs(vy_cmd) < 0.1:
                vy_cmd = 0
            if np.abs(vz_cmd) < 0.1:
                vz_cmd = 0
            self.y0 = self.y0 + vy_cmd * rate
            self.x0 = self.x0 + vx_cmd * rate
            self.z0 = self.z0 - vz_cmd * rate
        else:
            wx_cmd = right_joy[1]
            wy_cmd = right_joy[0]
            wz_cmd = left_joy[0]
            if np.abs(wx_cmd) < 0.1:
                wx_cmd = 0
            if np.abs(wy_cmd) < 0.1:
                wy_cmd = 0
            if np.abs(wz_cmd) < 0.1:
                wz_cmd = 0
            cmd = np.array([wx_cmd, wy_cmd, wz_cmd])
            omega_hat = np.array(
                [[0, -cmd[2], cmd[1]], [cmd[2], 0, -cmd[0]], [-cmd[1], cmd[0], 0]]
            )
            self.R0 = self.R0 @ (np.eye(3) + omega_hat / 100)

        # time.sleep(0.01)
        T_cmd = self.T0 @ np.vstack(
            [
                np.hstack(
                    [self.R0, np.array([self.x0, self.y0, self.z0]).reshape(3, 1)]
                ),
                np.array([0, 0, 0, 1]),
            ]
        )
        return T_cmd


class Follow_traj:
    def __init__(self, traj, T0, robot_model, link_name="link7"):
        """ "
        traj: (N, 3) with time first.
        """
        self.traj = traj
        self.idx = 0
        self.first_t = None

        self.fx = interpolate.interp1d(traj[:, 0], traj[:, 1])
        self.fy = interpolate.interp1d(traj[:, 0], traj[:, 2])

        self.Rref = T0[:3, :3].copy()
        self.pref = T0[:3, 3].copy()

        self.ps_t = []
        self.ps = []
        self.extra_time = 0.2
        self.link_name = link_name
        self.robot_model = robot_model

    def __call__(self, state):

        if self.first_t is None:
            self.first_t = time.time()

        elapsed_t = time.time() - self.first_t

        D = self.robot_model.getInfo(state.q, state.dq)
        p = D["poses"][self.link_name][:2, 3]

        self.ps.append(p)
        self.ps_t.append(elapsed_t)

        query_t = elapsed_t

        if elapsed_t > self.traj[-1, 0]:
            query_t = self.traj[-1, 0]

        # print("elapsed_t", elapsed_t)
        px = self.fx(query_t)
        py = self.fy(query_t)

        dp = np.array([px, py, 0])

        T_cmd = np.vstack(
            [
                np.hstack([self.Rref, (self.pref + dp).reshape(3, 1)]),
                np.array([0, 0, 0, 1]),
            ]
        )
        return T_cmd

    def applicable(self, s, t):
        return t < self.traj[-1, 0] + self.extra_time

    def goal_reached(self, s, t):
        # just run this until it is not applicable anymore
        return False


class DiffIKController:
    def __init__(
        self,
        model,
        T_cmd_fun=None,
        T_cmd=None,
        dt=0.01,
        ef_frame="link7",
        kp=0.5 * np.diag([10, 10, 10, 10, 10, 10]),
        kv=0.0 * np.diag([2, 2, 2, 2, 2, 2, 2]),
        joint_kv=1.5 * np.array([80.0, 80.0, 40.0, 40.0, 8.0, 8.0, 8.0]),
        joint_kp=2.0 * np.array([3000.0, 3000.0, 800.0, 800.0, 200.0, 200.0, 200.0]),
        k_reg=0.5 * np.diag([10, 10, 10, 10, 10, 10, 10]),
        dq_max=10,
        control_mode="velocity",
        max_error=0.05,
    ):
        if T_cmd_fun is None and T_cmd is None:
            raise Exception("Either T_cmd_fun or T_cmd must be provided!")

        # define solver
        self.model = model
        self.kp = kp
        self.kv = kv
        self.k_reg = k_reg
        self.ef_frame = ef_frame
        self.q0 = self.model.q0.reshape(7, 1).copy()
        self.T_cmd_fun = T_cmd_fun
        self.T_cmd = T_cmd
        self.dt = dt
        self.joint_kv = joint_kv
        self.joint_kp = joint_kp
        self.dq_max = dq_max
        self.max_error = max_error
        try:
            self.control_mode = {"velocity": 3, "torque": 2, "position": 1}[
                control_mode
            ]
        except:
            raise Exception("The selected control mode is not valid!")

    def __call__(self, q, dq, T_cmd):
        _q = np.array(q).reshape(7, 1)
        _dq = np.array(dq).reshape(7, 1)
        info = self.model.getInfo(_q, _dq)
        _T_current = info["poses"][self.ef_frame]
        _R_current = _T_current[0:3, 0:3]
        _t_current = _T_current[0:3, -1].reshape(3, 1)
        _R_cmd = T_cmd[0:3, 0:3]
        _t_cmd = T_cmd[0:3, -1].reshape(3, 1)

        J = info["Js"][self.ef_frame]
        J_inv = np.linalg.inv(J.T @ J + 1e-5 * np.eye(J.shape[1])) @ J.T
        P = np.eye(J.shape[1]) - J_inv @ J
        R_error = pin.log3(_R_cmd @ _R_current.T).reshape(3, 1)
        t_error = (_t_cmd - _t_current).reshape(3, 1)
        if np.linalg.norm(t_error) > self.max_error:
            raise Exception("The goal pose is too far!")
        pose_error = np.vstack([t_error, R_error])
        if np.linalg.det(J @ J.T) > 0.00001:
            dq_des = (
                J_inv @ (self.kp @ pose_error)
                + P @ self.k_reg @ (self.q0 - _q)
                - self.kv @ _dq
            ).squeeze()
        else:
            raise Exception("The robot is too close to singularity!")

        dq_des = np.clip(dq_des, -self.dq_max * np.ones(7), self.dq_max * np.ones(7))
        return dq_des

    def set_target(self, T_cmd):
        self.T_cmd = T_cmd

    def setup(self, s):
        # if self.T_cmd is None
        #     info = self.model.getInfo(np.array(s.q), np.array(s.dq))
        #     self.T_cmd = info["poses"][self.ef_frame]
        self.q_des = np.array(s.q)

    def get_control(self, state, t):
        if self.T_cmd is None:
            T_des = self.T_cmd_fun(state)
        else:
            T_des = self.T_cmd

        dq_des = self.__call__(np.array(state.q), np.array(state.dq), T_des)
        self.q_des += self.dt * dq_des
        if self.control_mode == 3:
            return FlexivCmd(dq=dq_des, mode=self.control_mode)
        else:
            return FlexivCmd(
                q=self.q_des,
                dq=dq_des,
                kp=self.joint_kp,
                kv=self.joint_kv,
                mode=self.control_mode,
            )

    def applicable(self, s, t):
        return self.T_cmd_fun.applicable(s, t)

    def goal_reached(self, s, t):
        return self.T_cmd_fun.goal_reached(s, t)
