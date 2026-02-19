#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

class MasterCalibrationTest(Node):
    def __init__(self):
        super().__init__('master_calibration_test')
        self.arm_publisher = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        
        self.get_logger().info('=============================================')
        self.get_logger().info('=== VERSIÓN ESTABLE PARA CALIBRACIÓN PnP ===')
        self.get_logger().info('=== OBJETIVO: POSICIÓN MAESTRA (X=0.20) ===')
        self.get_logger().info('=============================================')
        
        self.create_timer(5.0, self.timer_callback)
        self.step = 0

    def send_arm(self, positions, duration=3.0):
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start.sec = int(duration)
        traj.points.append(point)
        self.arm_publisher.publish(traj)

    def send_gripper(self, pos):
        goal = GripperCommand.Goal()
        goal.command.position = pos
        self.gripper_client.wait_for_server()
        self.gripper_client.send_goal_async(goal)

    def timer_callback(self):
        if self.step == 0:
            self.get_logger().info('1. Calibrando Gripper (Abierto)...')
            self.send_gripper(0.019)
            self.send_arm([0.0, -0.4, 0.2, 0.5], duration=2.0) 
            self.step = 1

        elif self.step == 1:
            self.get_logger().warn('2. Bajada a POSE MAESTRA (Alineación Perfecta)')
            # Los angulos que confirmaste como "completamente perfectos"
            # EE aprox: X=0.20, Z=0.03
            self.send_arm([0.0, 0.23, 0.35, 0.99], duration=3.0)
            self.step = 2

        elif self.step == 2:
            self.get_logger().info('3. METRICAS PARA CALIBRACIÓN:')
            self.get_logger().info(' - J1: 0.0 rad')
            self.get_logger().info(' - J2: 0.23 rad')
            self.get_logger().info(' - J3: 0.35 rad')
            self.get_logger().info(' - J4: 0.99 rad')
            self.get_logger().info(' - Posición Cubo en Gazebo: X=0.18, Y=0.0')
            self.send_gripper(-0.007)
            self.step = 3

        elif self.step == 3:
            self.get_logger().info('4. Intento de levantamiento estable...')
            self.send_arm([0.0, -0.4, 0.2, 0.3], duration=6.0)
            self.step = 4
            self.get_logger().info('=== POSE ESTABLE MANTENIDA. LISTO PARA CALIBRAR ===')

def main():
    rclpy.init()
    node = MasterCalibrationTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()