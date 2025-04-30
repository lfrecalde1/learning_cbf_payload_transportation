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
from cbf_polytopes import fancy_plots_3, plot_states_position, fancy_plots_4, plot_control_actions_reference, plot_angular_velocities
from cbf_polytopes import fancy_plots_1, plot_error_norm
import matplotlib.pyplot as plt

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

## Functions from casadi 
rotation_error_norm_f = rotation_matrix_error_norm_c()
rotation_error_f = rotation_matrix_error_c()

class PayloadDynamicsNode(Node):
    def __init__(self):
        super().__init__('PayloadDynamics')

        # Time Definition
        self.ts = 0.05
        self.final = 10
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

        # Computing explicit dynamics of the sytem only symbolic values
        self.payload_dynamics = self.system_dynamics()

        # Payload odometry
        self.odom_payload_msg = Odometry()
        self.publisher_payload_odom_ = self.create_publisher(Odometry, "odom", 10)

        # Position of the system
        pos_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
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
        self.N = 10
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
        linear_acceleration = (1/self.mass)*R@F - self.gravity*self.z

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
        linear_acceleration = (1/self.mass)*R@F - self.gravity*self.z

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
        kp_min = 25.0
        print(kp_min)
        print(kv_min)
        print(c1)
        print("--------------------------")

        ## Compute minimiun values for the angular controller
        eigenvalues = np.linalg.eigvals(self.inertia)
        min_eigenvalue = np.min(eigenvalues)
        c2 = 0.05
        kw_min = (1/2)*c2 + (1/4) + 0.1
        kr_min = c2*(kw_min*kw_min)/(min_eigenvalue*(4*(kw_min - (1/2)*c2) - 1))
        print(kr_min)
        print(kw_min)
        print(c2)
        print("--------------------------")

        ## Gains Constrol Actions
        R_force = 0.5*np.eye(3)
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
            angular_displacement_error = rotation_error_norm_f(quat_references, quat)
            angular_velocity_error = quad_angular_states - rotation_error_f(quat_references, quat)@quad_angular_state_references

            lyapunov_position_quad = (1/2)*kp_min*error_position_quad.T@error_position_quad + (1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad + c1*error_position_quad.T@error_velocity_quad
            lyapunov_orientation_quad = kr_min*angular_displacement_error.T@angular_displacement_error + (1/2)*angular_velocity_error.T@self.inertia@angular_velocity_error + c2*angular_displacement_error.T@angular_velocity_error


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
        quad_pos_state_references = ca.vertcat(1, 1, 1)

        # Quadrotor states velocity
        quad_vel_states = ca.vertcat(X[3, self.N], X[4, self.N], X[5, self.N])

        # Quadrotor states reference velocity
        quad_vel_state_references = ca.vertcat(0.0, 0.0, 0.0)

        # Quadrotor states quaternions
        quat = ca.vertcat(X[6, self.N], X[7, self.N], X[8, self.N], X[9, self.N])

        # Quadrotor states reference quaternions
        quat_references = ca.vertcat(1.0, 0.0, 0.0, 0.0)

        # Quadrotor states angular velocity
        quad_angular_states = ca.vertcat(X[10, self.N], X[11, self.N], X[12, self.N])
        # Quadrotor states reference angular velocity
        quad_angular_state_references = ca.vertcat(0.0, 0.0, 0.0)

        # Computing translational error of the sytem quadrotor
        error_position_quad = quad_pos_states - quad_pos_state_references
        error_velocity_quad = quad_vel_states - quad_vel_state_references

        # Angular error and angular velocity error
        angular_displacement_error = rotation_error_norm_f(quat_references, quat)
        angular_velocity_error = quad_angular_states - rotation_error_f(quat_references, quat)@quad_angular_state_references


        lyapunov_position_quad = (1/2)*kp_min*error_position_quad.T@error_position_quad + (1/2)*(self.mass)*error_velocity_quad.T@error_velocity_quad + c1*error_position_quad.T@error_velocity_quad
        lyapunov_orientation_quad = kr_min*angular_displacement_error.T@angular_displacement_error + (1/2)*angular_velocity_error.T@self.inertia@angular_velocity_error + c2*angular_displacement_error.T@angular_velocity_error
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
        F_min = ca.DM([-10, -10, 0])
        F_max = ca.DM([10, 10, 30])
        T_min = ca.DM([-0.03, -0.03, -0.03])
        T_max = ca.DM([0.03, 0.03, 0.03])

        v_min = ca.repmat(ca.vertcat(F_min, T_min), self.N, 1)
        v_max = ca.repmat(ca.vertcat(F_max, T_max), self.N, 1)

        lbx[n_states*(self.N+1):] = v_min
        ubx[n_states*(self.N+1):] = v_max

        args = {'lbg': ca.DM.zeros((n_states*(self.N+1), 1)), 'ubg': ca.DM.zeros((n_states*(self.N+1), 1)), 'lbx': lbx, 'ubx': ubx}

        return solver, args

    def send_odometry(self, x):
        position = x[0:3]
        quat = x[6:10]

        # Function that send odometry

        self.odom_payload_msg.header.frame_id = "world"
        self.odom_payload_msg.header.stamp = self.get_clock().now().to_msg()

        self.odom_payload_msg.pose.pose.position.x = position[0]
        self.odom_payload_msg.pose.pose.position.y = position[1]
        self.odom_payload_msg.pose.pose.position.z = position[2]

        self.odom_payload_msg.pose.pose.orientation.x = quat[1]
        self.odom_payload_msg.pose.pose.orientation.y = quat[2]
        self.odom_payload_msg.pose.pose.orientation.z = quat[3]
        self.odom_payload_msg.pose.pose.orientation.w = quat[0]

        # Send Messag
        self.publisher_payload_odom_.publish(self.odom_payload_msg)
        return None 

    def run(self):
        # Set the states to simulate
        x = np.zeros((self.n_x, self.t.shape[0] + 1 - self.N), dtype=np.double)
        u = np.zeros((self.n_u, self.t.shape[0] - self.N), dtype=np.double)

        x[:, 0] = self.x_0
        u[2, :] = (self.mass + 0.0001)*self.gravity

        # Check MPC
        solver, args = self.MPC()

        u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        u0[2, :] = self.mass*self.gravity
        X0 = ca.repmat(x[:, 0], 1, self.N+1)

        xd = np.zeros((self.n_x, self.t.shape[0] + 1 - self.N), dtype=np.double)
        xd[0, :] = 1.0
        xd[1, :] = 1.0
        xd[2, :] = 1.0
        
        xd[3, :] = 0.0
        xd[4, :] = 0.0
        xd[5, :] = 0.0

        xd[6, :] = 1.0
        xd[7, :] = 0.0
        xd[8, :] = 0.0
        xd[9, :] = 0.0
        
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
            self.send_odometry(x[:, k])

            ## Control
            args['x0'] = ca.vertcat(ca.reshape(X0, self.n_states*(self.N+1), 1), ca.reshape(u0, self.n_controls*self.N, 1))

            sol = solver(
                x0=args['x0'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
                p=args['p'])

            u_optimal = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
            X0 = ca.reshape(sol['x'][: self.n_states * (self.N+1)], self.n_states, self.N+1)


            # Dynamics of the system
            x_k = np.array(self.payload_dynamics(x[:, k], u_optimal[:, 0], self.ts))
            x[:, k + 1] = x_k.reshape((self.n_x, ))

            ## Update optimalk values
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
        #fig11, ax11, ax21, ax31 = fancy_plots_3()
        #plot_states_position(fig11, ax11, ax21, ax31, x[0:3, :], self.h_d[0:3, :], self.t, "Position of the System No drag")
        #plt.show()

        ## Control Actions
        #fig13, ax13, ax23, ax33, ax43 = fancy_plots_4()
        #plot_control_actions_reference(fig13, ax13, ax23, ax33, ax43, F, M, self.f_d, self.M_d, self.t, "Control Actions of the System No Drag")
        #plt.show()

        #fig14, ax14 = fancy_plots_1()
        #plot_error_norm(fig14, ax14, h_e, self.t, "Error Norm of the System No Drag")
        #plt.show()
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