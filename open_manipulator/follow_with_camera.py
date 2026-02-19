#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class ForcefulVisionFollower(Node):
    def __init__(self):
        super().__init__('forceful_vision_follower')
        self.arm_publisher = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.subscription_img = self.create_subscription(Image, '/static_camera/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()
        
        # Centro de la cámara (Pixel -> Robot)
        self.REF_PIXEL_X = 320 
        self.REF_PIXEL_Y = 315
        
        # Variables de objetivo detectado
        self.target_pixel = None
        self.robot_x = 0.18
        self.robot_y = 0.0
        
        # Estado del robot
        self.current_j1 = 0.0
        self.get_logger().info('=== SISTEMA DE SEGUIMIENTO DE ALTO TORQUE V32 ===')
        
        # Ciclo de control rápido (10Hz) para seguimiento fluido
        self.create_timer(0.1, self.control_loop)
        self.step = "TRACKING"

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Umbralización para detectar el objeto (oscuro sobre fondo claro)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        # ROI: Solo mirar la zona de trabajo (evitar ver el propio brazo arriba)
        mask = np.zeros_like(thresh)
        mask[250:, :] = 255
        thresh = cv2.bitwise_and(thresh, mask)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 200:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.target_pixel = (cx, cy)
                    
                    # Convertir pixel a coordenadas aproximadas del mundo robot
                    # Y frontal robot es la diferencia horizontal en imagen
                    self.robot_y = (self.REF_PIXEL_X - cx) * 0.0025
                    self.robot_x = 0.18 + (self.REF_PIXEL_Y - cy) * 0.0015
        else:
            self.target_pixel = None

    def send_arm(self, j1, j2, j3, j4, time_s=0.2):
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions = [float(j1), float(j2), float(j3), float(j4)]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(time_s * 1e9)
        traj.points.append(point)
        self.arm_publisher.publish(traj)

    def control_loop(self):
        if self.target_pixel is None:
            return

        # 1. Calcular ángulo J1 con ATAN2 (El más preciso)
        # Multiplicamos por un factor de corrección de perspectiva visual
        target_j1 = math.atan2(self.robot_y, self.robot_x) * 2.8 
        
        # 2. Calcular extensión
        dx = self.robot_x - 0.18
        target_j2 = 0.23 - (dx * 0.5)
        target_j3 = 0.35 + (dx * 0.5)
        
        if self.step == "TRACKING":
            # Seguimos al objeto suavemente
            self.send_arm(target_j1, target_j2, target_j3, 0.99)
            
            # Si el error es muy pequeño, podríamos decidir agarrar
            # Pero por ahora solo queremos ver que ROTE.
            # Imprimimos log cada segundo para no saturar
            if int(rclpy.clock.Clock().now().nanoseconds / 1e8) % 10 == 0:
                self.get_logger().info(f'Tracking: J1={target_j1:.3f} rad')

def main(args=None):
    rclpy.init(args=args)
    node = ForcefulVisionFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
