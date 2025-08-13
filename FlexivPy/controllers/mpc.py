import os
import pinocchio as pin
import crocoddyl
import pinocchio
import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
from FlexivPy import ASSETS_PATH

class UnconstrainedReachingMPC:
    def __init__(self, HORIZON=50, dt = 0.01, ee_frame_name = 'link7'):
        self.HORIZON = HORIZON
        self.max_iterations = 500
        self.dt = dt
        urdf = os.path.join(ASSETS_PATH, "flexiv_rizon10s_kinematics.urdf")
        meshes_dir = os.path.join(ASSETS_PATH, "meshes")
        self.pin_robot = RobotWrapper.BuildFromURDF(urdf, meshes_dir)
        self.pinRef = pin.LOCAL_WORLD_ALIGNED
        self.rmodel = self.pin_robot.model
        self.rdata = self.pin_robot.data
        self.efId = self.rmodel.getFrameId(ee_frame_name)
        self.ccdyl_state = crocoddyl.StateMultibody(self.rmodel)
        self.ccdyl_actuation = crocoddyl.ActuationModelFull(self.ccdyl_state)
        self.nu = self.ccdyl_actuation.nu
        self.running_models = []
        self.desired_poses = None
        self.solver = None

    def initialize(self, q0=np.array([-0.007916312664747238, -0.5759644508361816, 0.01036121603101492, 1.4920966625213623, -0.006107804365456104, 0.49659037590026855, 0.004393362440168858])):
        self.q0 = q0.copy()
        self.x0 =  np.concatenate([q0, np.zeros(self.rmodel.nv)])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        self.efPos0 = self.rdata.oMf[self.efId].translation
        self.efRot0 = self.rdata.oMf[self.efId].rotation
        self.xs = [self.x0]*(self.HORIZON + 1)
        self.createProblem()
        self.createSolver()
        self.us = self.solver.problem.quasiStatic([self.x0]*self.HORIZON) 

    def createProblem(self):
        for t in range(self.HORIZON+1):
            costModel = crocoddyl.CostModelSum(self.ccdyl_state, self.nu)
            # Control regularization cost
            uResidual = crocoddyl.ResidualModelControlGrav(self.ccdyl_state)
            uRegCost = crocoddyl.CostModelResidual(self.ccdyl_state, uResidual)
            # State regularization cost
            xResidual = crocoddyl.ResidualModelState(self.ccdyl_state, self.x0)
            xRegCost = crocoddyl.CostModelResidual(self.ccdyl_state, xResidual)
            # endeff frame translation cost
            endeff_translation = self.efPos0  
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.ccdyl_state, self.efId, endeff_translation)
            frameTranslationCost = crocoddyl.CostModelResidual(self.ccdyl_state, frameTranslationResidual)
            frameRotationResidual = crocoddyl.ResidualModelFrameRotation(self.ccdyl_state, self.efId, self.efRot0)
            frameRotationCost = crocoddyl.CostModelResidual(self.ccdyl_state, frameRotationResidual)

            if t < self.HORIZON:
                costModel.addCost("stateReg", xRegCost, 1e-1)
                costModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
                costModel.addCost("translation", frameTranslationCost, 30)
                costModel.addCost("rotation", frameRotationCost, 30)
            else:
                costModel.addCost("translation", frameTranslationCost, 30)
                costModel.addCost("rotation", frameRotationCost, 30)
                costModel.addCost("stateReg", xRegCost, 1e-1)
            dmodel = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.ccdyl_state, self.ccdyl_actuation, costModel)
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
            self.running_models += [model]
        self.ocp = crocoddyl.ShootingProblem(self.x0, self.running_models[:-1], self.running_models[-1])
        
    def createSolver(self):
        solver = crocoddyl.SolverFDDP(self.ocp)
        self.solver = solver

    def getSolution(self, k=None):
        if k is None: 
            x_idx = 1
            u_idx = 0
        else:
            x_idx = k
            u_idx = k
        q = self.xs[x_idx][:self.rmodel.nq]
        dq = self.xs[x_idx][self.rmodel.nq:]
        return dict(
            q = q,
            dq = dq, 
            tau = self.us[u_idx],
        )
    
    def updateAndSolve(self, q, dq):
        assert self.solver is not None, "Solver has not been initialized. Call initialize() first."
        # Shift the desired poses one step forward in time if they are set
        if self.desired_poses is not None:
            desired_poses = self.desired_poses[1:] + [self.desired_poses[-1]]
            self.setChunkPoseAction(desired_poses)
        q_ = q
        dq_ = dq
        pin.framesForwardKinematics(self.rmodel, self.rdata, q_)
        x = np.hstack([q_, dq_])
        self.solver.problem.x0 = x
        self.xs = list(self.solver.xs[1:]) + [self.solver.xs[-1]]
        self.xs[0] = x
        self.us = list(self.us[1:]) + [self.us[-1]] 
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()
    
    def solve(self):
        assert self.solver is not None, "Solver has not been initialized. Call initialize() first."
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()
    
    def setChunkPoseAction(self, desired_poses):
        assert self.solver is not None, "Solver has not been initialized. Call initialize() first."
        self.desired_poses = desired_poses
        assert len(desired_poses) == self.HORIZON+1, "The number of desired poses must match the horizon length."
        action_models = list(self.solver.problem.runningModels) + [self.solver.problem.terminalModel]
        for i, action_model in enumerate(action_models):
            rot_residual = action_model.differential.costs.costs["rotation"].cost.residual
            trans_residual = action_model.differential.costs.costs["translation"].cost.residual
            rot_ref = rot_residual.reference.copy()
            trans_ref = trans_residual.reference.copy()
            rot_ref[:] = desired_poses[i][:3, :3]
            trans_ref[:] = desired_poses[i][:3, 3]
            rot_residual.reference = rot_ref
            trans_residual.reference = trans_ref

    def getDesiredPoses(self):
        poses = []
        action_models = list(self.solver.problem.runningModels) + [self.solver.problem.terminalModel]
        for i, action_model in enumerate(action_models):
            rot_residual = action_model.differential.costs.costs["rotation"].cost.residual
            trans_residual = action_model.differential.costs.costs["translation"].cost.residual
            rot_ref = rot_residual.reference.copy()
            trans_ref = trans_residual.reference.copy()
            T = np.eye(4)
            T[:3, :3] = rot_ref
            T[:3, -1] = trans_ref
            poses.append(T)
        return poses

    def setSinglePoseAction(self, T_des):
        assert T_des.shape == (4, 4), "T_des must be a 4x4 transformation matrix."
        desired_poses = [T_des] * (self.HORIZON + 1)
        self.setChunkPoseAction(desired_poses)
    
    def setHomeCommand(self):
        R_des = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])
        t_des = np.array([0.70149498, -0.11319596,  0.62456593])
        T_des = np.eye(4)
        T_des[:3, :3] = R_des
        T_des[:3, 3] = t_des
        desired_poses = [T_des] * (self.HORIZON + 1)
        self.setChunkPoseAction(desired_poses)
        



# from dataclasses import dataclass
# import crocoddyl
# import pinocchio
# import numpy as np


# @dataclass
# class Mpc_cfg:
#     gripperPose_run: float = 1.0
#     gripperPose_terminal: float = 1e3
#     qReg: float = 1.0
#     vReg: float = 1.0
#     uReg: float = 1e-1
#     vel_terminal: float = 1e2
#     dt: float = 0.05
#     horizon: int = 20


# class Mpc_generator:
#     def __init__(self, robot, x0, Tdes, mpc_cfg: Mpc_cfg = Mpc_cfg()):
#         """

#         Notes:

#         After solving and optimization problem, you can access information with:
#         solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId("link7")].translation.T,




#         """

#         self.mpc_cfg = mpc_cfg
#         self.robot = robot
#         state = crocoddyl.StateMultibody(robot)
#         actuation = crocoddyl.ActuationModelFull(state)
#         x0 = x0

#         nu = 7

#         runningCostModel = crocoddyl.CostModelSum(state)
#         terminalCostModel = crocoddyl.CostModelSum(state)

#         # uResidual = crocoddyl.ResidualModelControlGrav(state, nu)
#         # uResidual = crocoddyl.ResidualModelJointEffort(state, nu)
#         uResidual = crocoddyl.ResidualModelJointAcceleration(state, nu)
#         xResidual = crocoddyl.ResidualModelState(state, x0, nu)

#         framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
#             state,
#             self.robot.getFrameId("link7"),
#             pinocchio.SE3(Tdes),
#             nu,
#         )

#         goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
#         # xRegCost = crocoddyl.CostModelResidual(state, xResidual)

#         velRegCost = crocoddyl.CostModelResidual(
#             state,
#             crocoddyl.ActivationModelWeightedQuad(
#                 np.concatenate([np.zeros(7), np.ones(7)])
#             ),
#             xResidual,
#         )

#         qRegCost = crocoddyl.CostModelResidual(
#             state,
#             crocoddyl.ActivationModelWeightedQuad(
#                 np.concatenate([np.ones(7), np.zeros(7)])
#             ),
#             xResidual,
#         )

#         uRegCost = crocoddyl.CostModelResidual(state, uResidual)

#         # Then let's added the running and terminal cost functions
#         runningCostModel.addCost(
#             "gripperPose", goalTrackingCost, mpc_cfg.gripperPose_run
#         )
#         runningCostModel.addCost("qReg", qRegCost, mpc_cfg.qReg)
#         runningCostModel.addCost("velReg", velRegCost, mpc_cfg.vReg)

#         runningCostModel.addCost("uReg", uRegCost, mpc_cfg.uReg)
#         terminalCostModel.addCost(
#             "gripperPose", goalTrackingCost, mpc_cfg.gripperPose_terminal
#         )
#         # terminalCostModel.addCost("velReg", velRegCost, mpc_cfg.vel_terminal)

#         runningModel = crocoddyl.IntegratedActionModelEuler(
#             crocoddyl.DifferentialActionModelFreeFwdDynamics(
#                 state, actuation, runningCostModel
#             ),
#             mpc_cfg.dt,
#         )
#         terminalModel = crocoddyl.IntegratedActionModelEuler(
#             crocoddyl.DifferentialActionModelFreeFwdDynamics(
#                 state, actuation, terminalCostModel
#             ),
#             0.0,
#         )

#         self.problem = crocoddyl.ShootingProblem(
#             x0, [runningModel] * mpc_cfg.horizon, terminalModel
#         )
#         self.solver = crocoddyl.SolverFDDP(self.problem)

#         self.solver.setCallbacks(
#             [
#                 crocoddyl.CallbackVerbose(),
#                 crocoddyl.CallbackLogger(),
#             ]
#         )

#     def update_x0(self, x0):
#         self.solver.problem.x0 = x0

#         # for k in range(self.solver.problem.T):
#         #     self.solver.problem.runningModels[k].differential.costs.costs[
#         #         "xReg"
#         #     ].cost.residual.reference = x0

#     def update_reg(self, qreg, vreg):
#         for k in range(self.solver.problem.T):
#             self.solver.problem.runningModels[k].differential.costs.costs[
#                 "qReg"
#             ].cost.residual.reference = qreg
#             self.solver.problem.runningModels[k].differential.costs.costs[
#                 "qReg"
#             ].cost.residual.reference = vreg

#     def update_Tdes(self, Tdes):
#         for k in range(self.solver.problem.T):
#             self.solver.prxRegoblem.runningModels[k].differential.costs.costs[
#                 "gripperPose"
#             ].cost.residual.reference = Tdes
#         self.problem.terminal_model.differential.costs.costs[
#             "gripperPose"
#         ].cost.residual.reference = Tdes
