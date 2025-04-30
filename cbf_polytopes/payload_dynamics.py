#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from casadi import Function
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import time
from cbf_polytopes import fancy_plots_3, plot_states_position, fancy_plots_4, plot_control_actions_reference, plot_angular_velocities, plot_states_quaternion, plot_control_actions_force, plot_control_actions_tau
from cbf_polytopes import fancy_plots_1, plot_error_norm
import matplotlib.pyplot as plt
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R


# Casadi Functions
def quatTorot_c(quat):
    # Normalized quaternion
    q = quat
    #q = q/(q.T@q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    Q = ca.vertcat(
        ca.horzcat(q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)),
        ca.horzcat(2*(q1*q2+q0*q3), q0**2+q2**2-q1**2-q3**2, 2*(q2*q3-q0*q1)),
        ca.horzcat(2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2+q3**2-q1**2-q2**2))

    return Q

def rotation_matrix_error_norm_c():
    # Desired Quaternion
    qd = ca.MX.sym('qd', 4, 1)

    # Current quaternion
    q = ca.MX.sym('q', 4, 1)

    Rd = quatTorot_c(qd)
    R = quatTorot_c(q)

    error_matrix = (Rd.T@R - R.T@Rd)/2

    vector_error = ca.vertcat(error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0])

    error_orientation_scalar_f = Function('error_orientation_scalar_f', [qd, q], [vector_error])

    return error_orientation_scalar_f

def rotation_matrix_error_c():
    # Desired Quaternion
    qd = ca.MX.sym('qd', 4, 1)

    # Current quaternion
    q = ca.MX.sym('q', 4, 1)

    Rd = quatTorot_c(qd)
    R = quatTorot_c(q)

    error_matrix = R.T@Rd
    error_matrix_f = Function('error_matrix_f', [qd, q], [error_matrix])

    return error_matrix_f

def cost_quaternion_casadi(qd, q):
    # Conjugate of desired quaternion
    qd_conjugate = ca.vertcat(qd[0], -qd[1], -qd[2], -qd[3])

    # Quaternion multiplication q_e = qd_conjugate * q
    H_r_plus = ca.vertcat(
        ca.horzcat(qd_conjugate[0], -qd_conjugate[1], -qd_conjugate[2], -qd_conjugate[3]),
        ca.horzcat(qd_conjugate[1], qd_conjugate[0], qd_conjugate[3], -qd_conjugate[2]),
        ca.horzcat(qd_conjugate[2], -qd_conjugate[3], qd_conjugate[0], qd_conjugate[1]),
        ca.horzcat(qd_conjugate[3], qd_conjugate[2], -qd_conjugate[1], qd_conjugate[0])
    )
    
    q_error = H_r_plus @ q

    ## Shortest path: if q_error[0] < 0, flip the sign
    condition = q_error[0] < 0
    q_error = ca.if_else(condition, -q_error, q_error)

    # Compute the angle and the log map
    norm_vec = ca.norm_2(q_error[1:4] + ca.np.finfo(np.float64).eps)
    angle = 2 * ca.atan2(norm_vec, q_error[0])

    # Avoid division by zero
    ln_quaternion = ca.vertcat(0.0, (1/2)*angle*q_error[1]/norm_vec, (1/2)*angle*q_error[2]/norm_vec, (1/2)*angle*q_error[3]/norm_vec)
    return  ln_quaternion

def hat_casadi(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0)
    )

def hat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

## Functions from casadi 
rotation_error_norm_f = rotation_matrix_error_norm_c()
rotation_error_f = rotation_matrix_error_c()

class PayloadDynamicsNode(Node):
    def __init__(self):
        super().__init__('PayloadDynamics')

        # Time Definition
        self.ts = 0.05
        self.final = 20
        self.t =np.arange(0, self.final + self.ts, self.ts, dtype=np.double)

        # Internal parameters defintion
        self.robot_num = 3
        self.mass = 0.2
        self.M_load = self.mass *  np.eye((3))
        self.inertia = np.array([[0.013344, 0.0, 0.0], [0.0, 0.012810, 0.0], [0.0, 0.0, 0.03064]], dtype=np.double)
        self.gravity = 9.81
        self.rho = np.array([[-0.288, 0.577, -0.288], [0.5, 0.0, -0.5], [0.01812, 0.01812, 0.01812]], dtype=np.double)
        self.cable_lenght = np.array([1.0, 1.0, 1.0], dtype=np.double)
        self.z = np.array([0.0, 0.0, 1.0], dtype=np.double)

        # Load shape parameters triangle
        self.p1 = np.array([0.15, 0.0, 0.0])
        self.p2 = np.array([-0.15, 0.2, 0.0])
        self.p3 = np.array([-0.15, -0.2, 0.0])
        self.p = np.vstack((self.p1, self.p2, self.p3)).T
        self.length = 1.5

        # Computing explicit dynamics of the sytem only symbolic values
        self.payload_dynamics = self.system_dynamics()

        # Payload odometry
        self.odom_payload_msg = Odometry()
        self.publisher_payload_odom_ = self.create_publisher(Odometry, "odom", 10)

        self.odom_payload_desired_msg = Odometry()
        self.publisher_payload_desired_odom_ = self.create_publisher(Odometry, "desired", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Position of the system
        pos_0 = np.array([0.0, 0.0, 2.0], dtype=np.double)
        # Linear velocity of the sytem respect to the inertial frame
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Angular velocity respect to the Body frame
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        # Initial Orientation expressed as quaternionn
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])

        # Auxiliary vector [x, v, q, w], which is used to update the odometry and the states of the system
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))
        self.n_x = self.x_0.shape[0]
        self.n_u = 6

        # MPC Parameters
        self.N = 20
        self.n_controls = self.n_u
        self.n_states = self.n_x

        self.timer = self.create_timer(self.ts, self.run)  # 0.01 seconds = 100 Hz
        self.start_time = time.time()

    def quatdot_c(self, quat, omega):
        # Quaternion evolution guaranteeing norm 1 (Improve this section)
        # INPUT
        # quat                                                   - actual quaternion
        # omega                                                  - angular velocities
        # OUTPUT
        # qdot                                                   - rate of change of the quaternion
        # Split values quaternion
        qw = quat[0, 0]
        qx = quat[1, 0]
        qy = quat[2, 0]
        qz = quat[3, 0]


        # Auxiliary variable in order to avoid numerical issues
        K_quat = 10
        quat_error = 1 - (qw**2 + qx**2 + qy**2 + qz**2)

        # Create skew matrix
        H_r_plus = ca.vertcat(ca.horzcat(quat[0, 0], -quat[1, 0], -quat[2, 0], -quat[3, 0]),
                                    ca.horzcat(quat[1, 0], quat[0, 0], -quat[3, 0], quat[2, 0]),
                                    ca.horzcat(quat[2, 0], quat[3, 0], quat[0, 0], -quat[1, 0]),
                                    ca.horzcat(quat[3, 0], -quat[2, 0], quat[1, 0], quat[0, 0]))

        omega_quat = ca.vertcat(0.0, omega[0, 0], omega[1, 0], omega[2, 0])


        q_dot = (1/2)*(H_r_plus@omega_quat) + K_quat*quat_error*quat
        return q_dot

    def quatTorot_c(self, quat):
        # Function to transform a quaternion to a rotational matrix
        # INPUT
        # quat                                                       - unit quaternion
        # OUTPUT                                     
        # R                                                          - rotational matrix

        # Normalized quaternion
        q = quat

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        Q = ca.vertcat(
            ca.horzcat(q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)),
            ca.horzcat(2*(q1*q2+q0*q3), q0**2+q2**2-q1**2-q3**2, 2*(q2*q3-q0*q1)),
            ca.horzcat(2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2+q3**2-q1**2-q2**2))

        # Compute Rotational Matrix
        #R = ca.MX.eye(3) + 2 * (q_hat@q_hat) + 2 * q[0] * q_hat
        R = Q
        return R

    def system_dynamics(self):
        # symbolic ts
        ts = ca.MX.sym('ts')

        # Sparse matrix for the inertia
        sparsity_I = ca.Sparsity.diag(3)

        # Create symbolic matrix with that sparsity
        I_sym = ca.MX(sparsity_I)

        # Assign the values from I_load
        I_sym[0, 0] = self.inertia[0, 0]
        I_sym[1, 1] = self.inertia[1, 1]
        I_sym[2, 2] = self.inertia[2, 2]

        #define input variables
        f1 = ca.MX.sym("f1")
        f2 = ca.MX.sym("f2")
        f3 = ca.MX.sym("f3")
        F = ca.vertcat(f1, f2, f3)              #force in Inertial frame

        tau_1 = ca.MX.sym("tau_1")
        tau_2 = ca.MX.sym("tau_2")
        tau_3 = ca.MX.sym("tau_3")
        tau = ca.vertcat(tau_1, tau_2, tau_3)              #moments in Payload frame

        # Complete control action with null space operator, we can also separe this 
        u = ca.vertcat(f1, f2, f3, tau_1, tau_2, tau_3)

        #position 
        p_x = ca.MX.sym('p_x')
        p_y = ca.MX.sym('p_y')
        p_z = ca.MX.sym('p_z')
        x_p = ca.vertcat(p_x, p_y, p_z)
    
        #linear vel
        vx_p = ca.MX.sym("vx_p")
        vy_p = ca.MX.sym("vy_p")
        vz_p = ca.MX.sym("vz_p")   
        v_p = ca.vertcat(vx_p, vy_p, vz_p)

        #angles quaternion 
        qw = ca.MX.sym('qw')
        qx = ca.MX.sym('qx')
        qy = ca.MX.sym('qy')
        qz = ca.MX.sym('qz')        
        quat = ca.vertcat(qw, qx, qy, qz)

        #angular velocity
        wx = ca.MX.sym('wx')
        wy = ca.MX.sym('wy',)
        wz = ca.MX.sym('wz')

        omega = ca.vertcat(wx, wy, wz) 

        # Full states of the system
        x = ca.vertcat(x_p, v_p, quat, omega)

        # Rotation matrix
        R = self.quatTorot_c(quat)
        cc_forces = ca.cross(omega, I_sym@omega)               #colaris and centripetel forces 

        # Linear Dynamics
        linear_velocity = v_p
        linear_acceleration = (1/self.mass)*F - self.gravity*self.z

        # Angular Dynamics
        quat_dt = self.quatdot_c(quat, omega)
        omgega_dot = - ca.inv(I_sym)@(cc_forces) + ca.inv(I_sym)@tau

        f_expl = ca.vertcat(linear_velocity,
                            linear_acceleration,
                            quat_dt,
                            omgega_dot
                            )
        dynamics_casadi_f = Function('dynamics_casadi_f',[x, u], [f_expl])

        ## Integration method
        k1 = dynamics_casadi_f(x, u)
        k2 = dynamics_casadi_f(x + (1/2)*ts*k1, u)
        k3 = dynamics_casadi_f(x + (1/2)*ts*k2, u)
        k4 = dynamics_casadi_f(x + ts*k3, u)

        # Compute forward Euler method
        xk = x + (1/6)*ts*(k1 + 2*k2 + 2*k3 + k4)
        casadi_kutta = Function('casadi_kutta',[x, u, ts], [xk]) 

        return casadi_kutta
    
    def publish_transforms(self, payload, wrench):
        tf_world_load = TransformStamped()
        tf_world_load.header.stamp = self.get_clock().now().to_msg()
        tf_world_load.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load.child_frame_id = 'payload'          # <-- imu_link is rotated
        tf_world_load.transform.translation.x = payload[0]
        tf_world_load.transform.translation.y = payload[1]
        tf_world_load.transform.translation.z = payload[2]
        tf_world_load.transform.rotation.x = payload[7]
        tf_world_load.transform.rotation.y = payload[8]
        tf_world_load.transform.rotation.z = payload[9]
        tf_world_load.transform.rotation.w = payload[6]
## --------------------------------------------------------------------------------------------------------------------
        ro = self.ro_w(payload)
        tf_world_p1 = TransformStamped()
        tf_world_p1.header.stamp = self.get_clock().now().to_msg()
        tf_world_p1.header.frame_id = 'payload'
        tf_world_p1.child_frame_id = 'p1'
        tf_world_p1.transform.translation.x = self.p1[0]
        tf_world_p1.transform.translation.y = self.p1[1]
        tf_world_p1.transform.translation.z = self.p1[2]

        tf_world_p2 = TransformStamped()
        tf_world_p2.header.stamp = self.get_clock().now().to_msg()
        tf_world_p2.header.frame_id = 'payload'
        tf_world_p2.child_frame_id = 'p2'
        tf_world_p2.transform.translation.x = self.p2[0]
        tf_world_p2.transform.translation.y = self.p2[1]
        tf_world_p2.transform.translation.z = self.p2[2]

        tf_world_p3 = TransformStamped()
        tf_world_p3.header.stamp = self.get_clock().now().to_msg()
        tf_world_p3.header.frame_id = 'payload'
        tf_world_p3.child_frame_id = 'p3'
        tf_world_p3.transform.translation.x = self.p3[0]
        tf_world_p3.transform.translation.y = self.p3[1]
        tf_world_p3.transform.translation.z = self.p3[2]

## ----------------------------------------------------------------------------------------------------
        quadrotors = self.quadrotors_w(payload, wrench)
        tf_world_q1 = TransformStamped()
        tf_world_q1.header.stamp = self.get_clock().now().to_msg()
        tf_world_q1.header.frame_id = 'world'
        tf_world_q1.child_frame_id = 'quadrotor_1'
        tf_world_q1.transform.translation.x = quadrotors[0, 0]
        tf_world_q1.transform.translation.y = quadrotors[1, 0]
        tf_world_q1.transform.translation.z = quadrotors[2, 0]

        tf_world_q2 = TransformStamped()
        tf_world_q2.header.stamp = self.get_clock().now().to_msg()
        tf_world_q2.header.frame_id = 'world'
        tf_world_q2.child_frame_id = 'quadrotor_2'
        tf_world_q2.transform.translation.x = quadrotors[0, 1]
        tf_world_q2.transform.translation.y = quadrotors[1, 1]
        tf_world_q2.transform.translation.z = quadrotors[2, 1]

        tf_world_q3 = TransformStamped()
        tf_world_q3.header.stamp = self.get_clock().now().to_msg()
        tf_world_q3.header.frame_id = 'world'
        tf_world_q3.child_frame_id = 'quadrotor_3'
        tf_world_q3.transform.translation.x = quadrotors[0, 2]
        tf_world_q3.transform.translation.y = quadrotors[1, 2]
        tf_world_q3.transform.translation.z = quadrotors[2, 2]

        tf_p1_q1 = TransformStamped()
        tf_p1_q1.header.stamp = self.get_clock().now().to_msg()
        tf_p1_q1.header.frame_id = 'p1'
        tf_p1_q1.child_frame_id = 'quadrotor_1_aux'
        aux = -(wrench[:, 0]/np.linalg.norm(wrench[:, 0]))*self.length
        tf_p1_q1.transform.translation.x = aux[0]
        tf_p1_q1.transform.translation.y = aux[1]
        tf_p1_q1.transform.translation.z = aux[2]

        ## Broadcast both transforms
        self.tf_broadcaster.sendTransform([tf_world_load, tf_world_q1, tf_world_q2, tf_world_q3, tf_world_p1, tf_world_p2, tf_world_p3])
        return None
    def ro_w(self, payload):
        # Rotation payload
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        # Translation
        t = payload[0:3]

        ro = np.zeros((3, self.p.shape[1]))

        for k in range(0, self.p.shape[1]):
            ro[:, k] = t + R_ql@self.p[:, k]
        return ro

    def quadrotors_w(self, payload, tension):
        # Rotation payload
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        # Translation
        t = payload[0:3]

        quat = np.zeros((3, self.p.shape[1]))

        for k in range(0, self.p.shape[1]):
            quat[:, k] = t + R_ql@self.p[:, k] + (tension[:, k]/np.linalg.norm(tension[:, k]))*self.length
        return quat
    
    def jacobian_forces(self, wrench, payload):
        I = np.eye(3)
        top_block = np.hstack([I, I, I])  # shape: (3, 9)

        # Block 2: three rotation matrices
        q = np.array([payload[7], payload[8], payload[9], payload[6]])
        R_object = R.from_quat(q)
        R_ql = R_object.as_matrix()

        p1_hat = hat(self.p1)
        p2_hat = hat(self.p2)
        p3_hat = hat(self.p3)

        bottom_block = np.hstack([p1_hat@R_ql.T, p2_hat@R_ql.T, p3_hat@R_ql.T])  # shape: (3, 9)

        # Final 6x9 matrix
        P = np.vstack([top_block, bottom_block])

        #
        tension = np.linalg.pinv(P)@wrench
        tensions_vectors = tension.reshape(-1, 3).T
        return tensions_vectors



    def MPC(self):
        # Sparse matrix for the inertia
        sparsity_I = ca.Sparsity.diag(3)

        # Create symbolic matrix with that sparsity
        I_sym = ca.MX(sparsity_I)

        # Assign the values from I_load
        I_sym[0, 0] = self.inertia[0, 0]
        I_sym[1, 1] = self.inertia[1, 1]
        I_sym[2, 2] = self.inertia[2, 2]

        #define input variables
        f1 = ca.MX.sym("f1")
        f2 = ca.MX.sym("f2")
        f3 = ca.MX.sym("f3")
        F = ca.vertcat(f1, f2, f3)              #force in Inertial frame

        tau_1 = ca.MX.sym("tau_1")
        tau_2 = ca.MX.sym("tau_2")
        tau_3 = ca.MX.sym("tau_3")
        tau = ca.vertcat(tau_1, tau_2, tau_3)              #moments in Payload frame

        # Complete control action with null space operator, we can also separe this 
        u = ca.vertcat(f1, f2, f3, tau_1, tau_2, tau_3)
        n_controls = u.numel()

        #position 
        p_x = ca.MX.sym('p_x')
        p_y = ca.MX.sym('p_y')
        p_z = ca.MX.sym('p_z')
        x_p = ca.vertcat(p_x, p_y, p_z)
    
        #linear vel
        vx_p = ca.MX.sym("vx_p")
        vy_p = ca.MX.sym("vy_p")
        vz_p = ca.MX.sym("vz_p")   
        v_p = ca.vertcat(vx_p, vy_p, vz_p)

        #angles quaternion 
        qw = ca.MX.sym('qw')
        qx = ca.MX.sym('qx')
        qy = ca.MX.sym('qy')
        qz = ca.MX.sym('qz')        
        quat = ca.vertcat(qw, qx, qy, qz)

        #angular velocity
        wx = ca.MX.sym('wx')
        wy = ca.MX.sym('wy',)
        wz = ca.MX.sym('wz')

        omega = ca.vertcat(wx, wy, wz) 

        # Full states of the system
        x = ca.vertcat(x_p, v_p, quat, omega)
        n_states = x.numel()

        # Rotation matrix
        R = self.quatTorot_c(quat)
        cc_forces = ca.cross(omega, I_sym@omega)               #colaris and centripetel forces 

        # Linear Dynamics
        linear_velocity = v_p
        linear_acceleration = (1/self.mass)*F - self.gravity*self.z

        # Angular Dynamics
        quat_dt = self.quatdot_c(quat, omega)
        omgega_dot = - ca.inv(I_sym)@(cc_forces) + ca.inv(I_sym)@tau

        # Nonlinear Funcitons
        f_expl = ca.vertcat(linear_velocity,
                            linear_acceleration,
                            quat_dt,
                            omgega_dot
                            )

        # Nonlinear Dynamics 
        dynamics_casadi_f = Function('dynamics_casadi_f',[x, u], [f_expl])

        # Create Symbolic values of the predicitrons and control actions
        X = ca.MX.sym('X', n_states, self.N + 1)
        U = ca.MX.sym('U', n_controls, self.N)
        P = ca.MX.sym('P', n_states + n_states)

        # Gains Definition
        # Compute Constant values for the trasnlational controller
        c1 = 1
        kv_min = c1 + 1/4 + 0.1
        kp_min = (c1*(kv_min*kv_min) + 2*kv_min*c1 - c1*c1)/((self.mass)*(4*(kv_min - c1)-1))
        kp_min = 80
        print(kp_min)
        print(kv_min)
        print(c1)
        print("--------------------------")

        ## Compute minimiun values for the angular controller
        eigenvalues = np.linalg.eigvals(self.inertia)
        min_eigenvalue = np.min(eigenvalues)
        c2 = 0.2
        kw_min = (1/2)*c2 + (1/4) + 0.1
        kr_min = c2*(kw_min*kw_min)/(min_eigenvalue*(4*(kw_min - (1/2)*c2) - 1))
        print(kr_min)
        print(kw_min)
        print(c2)
        print("--------------------------")

        ## Gains Constrol Actions
        R_force = 10*np.eye(3)
        R_torque = 1*np.eye(3)

        # Cost initial Value
        cost_fn = 0
        

        g = X[:, 0] - P[:n_states]  # constraints in the equation

        # Evolution of the system over the predictions
        for k in range(self.N):
            # Desired Vector
            xd = P[n_states:]

            quad_pos_states = ca.vertcat(X[0, k], X[1, k], X[2, k])
            # Quadrotor states reference position
            quad_pos_state_references = xd[0:3]

            # Quadrotor states velocity
            quad_vel_states = ca.vertcat(X[3, k], X[4, k], X[5, k])

            # Quadrotor states reference velocity
            quad_vel_state_references = xd[3:6]

            # Quadrotor states quaternions
            quat = ca.vertcat(X[6, k], X[7, k], X[8, k], X[9, k])

            # Quadrotor states reference quaternions
            quat_references = xd[6:10]

            # Quadrotor states angular velocity
            quad_angular_states = ca.vertcat(X[10, k], X[11, k], X[12, k])
            # Quadrotor states reference angular velocity
            quad_angular_state_references = xd[10:13]

            # Computing translational error of the sytem quadrotor
            error_position_quad = quad_pos_states - quad_pos_state_references
            error_velocity_quad = quad_vel_states - quad_vel_state_references

            # Angular error and angular velocity error
            angular_displacement_error = cost_quaternion_casadi(quat_references, quat)
            angular_velocity_error = quad_angular_states - rotation_error_f(quat_references, quat)@quad_angular_state_references

            lyapunov_position_quad = (1/2)*kp_min*error_position_quad.T@error_position_quad + (1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad + c1*error_position_quad.T@error_velocity_quad
            lyapunov_orientation_quad = kr_min*angular_displacement_error.T@angular_displacement_error + (1/2)*angular_velocity_error.T@self.inertia@angular_velocity_error


            # Control Action Desired
            force_desired = ca.vertcat(0.0, 0.0, self.mass*self.gravity)
            torque_desired = ca.vertcat(0.0, 0.0, 0.0)

            force_real = U[0:3, k]
            torque_real = U[3:6, k]

            # Deviation from the nominal values
            error_force = force_desired - force_real
            error_torque = torque_desired - torque_real

            # Cost Fucntion
            cost_fn = cost_fn + lyapunov_position_quad + lyapunov_orientation_quad + error_force.T@R_force@error_force + error_torque.T@R_torque@error_torque


            st_next = X[:, k+1]
            ## Integration method
            k1 = dynamics_casadi_f(X[:, k], U[:, k])
            k2 = dynamics_casadi_f(X[:, k] + (1/2)*self.ts*k1, U[:, k])
            k3 = dynamics_casadi_f(X[:, k] + (1/2)*self.ts*k2, U[:, k])
            k4 = dynamics_casadi_f(X[:, k] + self.ts*k3, U[:, k])

            # Compute forward Euler method
            st_next_RK4 = X[:, k] + (1/6)*self.ts*(k1 + 2*k2 + 2*k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

        # TERMINAL COST
        quad_pos_states = ca.vertcat(X[0, self.N], X[1, self.N], X[2, self.N])
        # Quadrotor states reference position
        quad_pos_state_references = xd[0:3]

        # Quadrotor states velocity
        quad_vel_states = ca.vertcat(X[3, self.N], X[4, self.N], X[5, self.N])

        # Quadrotor states reference velocity
        quad_vel_state_references = xd[3:6]

        # Quadrotor states quaternions
        quat = ca.vertcat(X[6, self.N], X[7, self.N], X[8, self.N], X[9, self.N])

        # Quadrotor states reference quaternions
        quat_references = xd[6:10]

        # Quadrotor states angular velocity
        quad_angular_states = ca.vertcat(X[10, self.N], X[11, self.N], X[12, self.N])

        # Quadrotor states reference angular velocity
        quad_angular_state_references = xd[10:13]

        # Computing translational error of the sytem quadrotor
        error_position_quad = quad_pos_states - quad_pos_state_references
        error_velocity_quad = quad_vel_states - quad_vel_state_references

        # Angular error and angular velocity error
        angular_displacement_error = cost_quaternion_casadi(quat_references, quat)
        angular_velocity_error = quad_angular_states - rotation_error_f(quat_references, quat)@quad_angular_state_references


        lyapunov_position_quad = (1/2)*kp_min*error_position_quad.T@error_position_quad + (1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad + c1*error_position_quad.T@error_velocity_quad
        lyapunov_orientation_quad = kr_min*angular_displacement_error.T@angular_displacement_error + (1/2)*angular_velocity_error.T@self.inertia@angular_velocity_error
        cost_fn = cost_fn + lyapunov_position_quad + lyapunov_orientation_quad

        # Reshape optimization variable
        OPT_variables = ca.vertcat(X.reshape((-1, 1)),U.reshape((-1, 1)))

        # Structure Optimization problem
        nlp_prob = {'f': cost_fn, 'x': OPT_variables, 'g': g, 'p': P}
        opts = {'ipopt': {'max_iter': 2000, 'print_level': 0, 'acceptable_tol': 1e-8, 'acceptable_obj_change_tol': 1e-6}, 'print_time': 0}

        # Optimization problem to be solved
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)


        # Constraints of the system
        lbx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))
        ubx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))

        ### ----------------------------------------------------------------- Lower contraints -------------------------------------
        lbx[0: n_states*(self.N+1): n_states] = -ca.inf     # X lower bound
        lbx[1: n_states*(self.N+1): n_states] = -ca.inf     # Y lower bound
        lbx[2: n_states*(self.N+1): n_states] = -0.1        # Z lower bound

        lbx[3: n_states*(self.N+1): n_states] = -10         # Vx lower bound
        lbx[4: n_states*(self.N+1): n_states] = -10         # Vy lower bound
        lbx[5: n_states*(self.N+1): n_states] = -10         # Vz lower bound

        # Quaternions upper bound constraint be careful
        lbx[6: n_states*(self.N+1): n_states] = -10         # qw lower bound
        lbx[7: n_states*(self.N+1): n_states] = -10         # qx lower bound
        lbx[8: n_states*(self.N+1): n_states] = -10         # qy lower bound
        lbx[9: n_states*(self.N+1): n_states] = -10         # qz lower bound

        lbx[10: n_states*(self.N+1): n_states] = -6         # Wx lower bound
        lbx[11: n_states*(self.N+1): n_states] = -6         # Wy lower bound
        lbx[12: n_states*(self.N+1): n_states] = -6         # Wz lower bound

        ### ----------------------------------------------------------------- Upper contraints -------------------------------------
        ubx[0: n_states*(self.N+1): n_states] = ca.inf      # X upper bound
        ubx[1: n_states*(self.N+1): n_states] = ca.inf      # Y upper bound
        ubx[2: n_states*(self.N+1): n_states] = 20          # Z upper bound

        ubx[3: n_states*(self.N+1): n_states] = 10          # Vx upper bound
        ubx[4: n_states*(self.N+1): n_states] = 10          # Vy upper bound
        ubx[5: n_states*(self.N+1): n_states] = 10          # Vz upper bound

        ubx[6: n_states*(self.N+1): n_states] = 10          # qw upper bound
        ubx[7: n_states*(self.N+1): n_states] = 10          # qx upper bound
        ubx[8: n_states*(self.N+1): n_states] = 10          # qy upper bound
        ubx[9: n_states*(self.N+1): n_states] = 10          # qz upper bound

        ubx[10: n_states*(self.N+1): n_states] = 6          # Wx upper bound
        ubx[11: n_states*(self.N+1): n_states] = 6          # Wy upper bound
        ubx[12: n_states*(self.N+1): n_states] = 6          # Wz upper bound


        # Constrainst control actions 
        F_min = ca.DM([-5, -5, 0])
        F_max = ca.DM([5, 5, 30])
        T_min = ca.DM([-0.1, -0.1, -0.1])
        T_max = ca.DM([0.1, 0.1, 0.1])

        v_min = ca.repmat(ca.vertcat(F_min, T_min), self.N, 1)
        v_max = ca.repmat(ca.vertcat(F_max, T_max), self.N, 1)

        lbx[n_states*(self.N+1):] = v_min
        ubx[n_states*(self.N+1):] = v_max

        args = {'lbg': ca.DM.zeros((n_states*(self.N+1), 1)), 'ubg': ca.DM.zeros((n_states*(self.N+1), 1)), 'lbx': lbx, 'ubx': ubx}

        return solver, args

    def send_odometry(self, x, odom_payload_msg, publisher_payload_odom):
        position = x[0:3]
        quat = x[6:10]

        # Function that send odometry

        odom_payload_msg.header.frame_id = "world"
        odom_payload_msg.header.stamp = self.get_clock().now().to_msg()

        odom_payload_msg.pose.pose.position.x = position[0]
        odom_payload_msg.pose.pose.position.y = position[1]
        odom_payload_msg.pose.pose.position.z = position[2]

        odom_payload_msg.pose.pose.orientation.x = quat[1]
        odom_payload_msg.pose.pose.orientation.y = quat[2]
        odom_payload_msg.pose.pose.orientation.z = quat[3]
        odom_payload_msg.pose.pose.orientation.w = quat[0]

        # Send Messag
        publisher_payload_odom.publish(odom_payload_msg)
        return None 

    def run(self):
        # Set the states to simulate
        x = np.zeros((self.n_x, self.t.shape[0] + 1 - self.N), dtype=np.double)
        u = np.zeros((self.n_u, self.t.shape[0] - self.N), dtype=np.double)

        x[:, 0] = self.x_0

        # Check MPC
        solver, args = self.MPC()

        u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        u0[2, :] = self.mass*self.gravity
        X0 = ca.repmat(x[:, 0], 1, self.N+1)

        # Desired states
        xd = np.zeros((self.n_x, self.t.shape[0] + 1 - self.N), dtype=np.double)
        xd[0, :] = 1.0
        xd[1, :] = 1.0
        xd[2, :] = 3.0
        
        xd[3, :] = 0.0
        xd[4, :] = 0.0
        xd[5, :] = 0.0

        theta1 = 1*np.pi/2
        n1 = np.array([0.0, 0.0, 1.0])
        qd = np.concatenate(([np.cos(theta1 / 2)], np.sin(theta1 / 2) * n1))

        xd[6, :] = qd[0]
        xd[7, :] = qd[1]
        xd[8, :] = qd[2]
        xd[9, :] = qd[3]
        
        xd[10, :] = 0.0
        xd[11, :] = 0.0
        xd[12, :] = 0.0
        
        # Simulation loop
        for k in range(0, self.t.shape[0] - self.N):
            # Get model
            tic = time.time()
            # Set init States optimizer
            args['p'] = ca.vertcat(x[:, k], xd[:, k])
            # Send Odometry
            self.send_odometry(x[:, k], self.odom_payload_msg, self.publisher_payload_odom_)
            self.send_odometry(xd[:, k], self.odom_payload_desired_msg, self.publisher_payload_desired_odom_)

            ## Control
            args['x0'] = ca.vertcat(ca.reshape(X0, self.n_states*(self.N+1), 1), ca.reshape(u0, self.n_controls*self.N, 1))

            # Set up states for optimal control
            sol = solver(
                x0=args['x0'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
                p=args['p'])

            # Get optimal control values and states
            u_optimal = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
            X0 = ca.reshape(sol['x'][: self.n_states * (self.N+1)], self.n_states, self.N+1)
            u[:, k] = np.array(u_optimal[:, 0]).reshape((6,))

            # Publish frames
            tension_vector = self.jacobian_forces(u[:, k], x[:, k])
            self.publish_transforms(x[:, k], tension_vector)



            # Dynamics of the system
            x_k = np.array(self.payload_dynamics(x[:, k], u[:, k], self.ts))
            x[:, k + 1] = x_k.reshape((self.n_x, ))

            ## Update optimal values
            u0 = ca.horzcat(u_optimal[:, 1:],ca.reshape(u_optimal[:, -1], -1, 1))
            X0 = ca.horzcat(X0[:, 1:],ca.reshape(X0[:, -1], -1, 1))

            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            self.get_logger().info(f"Sample time: {toc:.6f} seconds")
            self.get_logger().info(f"time: {self.t[k]:.6f} seconds")
            self.get_logger().info("PAYLOAD DYNAMICS")

        # Results of the system
        fig11, ax11, ax21, ax31 = fancy_plots_3()
        plot_states_position(fig11, ax11, ax21, ax31, x[0:3, :], xd[0:3, :], self.t, "Position of the System No drag")
        plt.show()

        fig12, ax12, ax22, ax32, ax42 = fancy_plots_4()
        plot_states_quaternion(fig12, ax12, ax22, ax32, ax42, x[6:10, :], xd[6:10, :], self.t, "Quaternions of the System No drag")
        plt.show()

        fig13, ax13, ax23, ax33 = fancy_plots_3()
        plot_control_actions_force(fig13, ax13, ax23, ax33, u[0:3, :], self.t, "Force Control Action of the System No drag")
        plt.show()

        fig14, ax14, ax24, ax34 = fancy_plots_3()
        plot_control_actions_tau(fig14, ax14, ax24, ax34, u[3:6, :], self.t, "Torque Control Action of the System No drag")
        plt.show()
        return None

def main(arg = None):
    rclpy.init(args=arg)
    payload_node = PayloadDynamicsNode()
    try:
        rclpy.spin(payload_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        payload_node.get_logger().info('Simulation stopped manually.')
        payload_node.destroy_node()
        rclpy.shutdown()
    finally:
        payload_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()