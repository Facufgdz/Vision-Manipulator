#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2
import numpy as np
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        self.br = CvBridge()
        
        # Suscripción a la cámara
        self.subscription = self.create_subscription(
            Image,
            '/static_camera/image_raw',
            self.image_callback,
            10)
        
        # Publicador para el brazo
        self.arm_publisher = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10)
        
        # Cliente de acción para el gripper
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        
        self.state = 'IDLE' # IDLE, DETECTING, MOVING_ABOVE, LOWERING, GRASPING, LIFTING
        self.object_x = 0.0
        self.object_y = 0.0
        
        self.get_logger().info('Nodo de Pick and Place iniciado.')
        
        # Timer para ejecutar la lógica de estados
        self.timer = self.create_timer(1.0, self.control_loop)

    def image_callback(self, msg):
        if self.state != 'IDLE' and self.state != 'DETECTING':
            return

        # Convertir a OpenCV
        frame = self.br.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Rango para color verde (ajustado para mayor sensibilidad)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Mostrar las ventanas de depuración
        cv2.imshow("Camara Original", frame)
        cv2.imshow("Mascara Verde", mask)
        cv2.waitKey(1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                M = cv2.moments(largest_contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                # Corregimos el mapeo: 
                # Con la cámara rotada 180º, cy más pequeño (arriba) es X más grande.
                # El centro (320, 240) es X=0.4.
                self.object_y = (320 - cx) * 0.0018
                self.object_x = 0.4 - (cy - 240) * 0.0018
                
                self.get_logger().info(f'Detección (Corregida): X={self.object_x:.3f}, Y={self.object_y:.3f}')
                self.state = 'DETECTING'

    def send_arm_command(self, positions, duration=2.0):
        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        traj.points.append(point)
        self.arm_publisher.publish(traj)

    def send_gripper_command(self, position):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position # 0.0 abierto? depende del modelo
        goal_msg.command.max_effort = -1.0
        self.gripper_client.wait_for_server()
        return self.gripper_client.send_goal_async(goal_msg)

    def control_loop(self):
        if self.state == 'DETECTING':
            self.get_logger().info('Pasando a mover encima del objeto...')
            # Abrir gripper primero
            self.send_gripper_command(0.019) 
            # Posición de aproximación (más inclinada hacia adelante)
            j1 = np.arctan2(self.object_y, self.object_x)
            self.send_arm_command([j1, 0.0, -0.5, 0.8])
            self.state = 'MOVING_ABOVE'
            
        elif self.state == 'MOVING_ABOVE':
            # Baja al NIVEL DEL SUELO pero sin chocar (Suavizado)
            self.get_logger().info('Bajando brazo con cuidado...')
            j1 = np.arctan2(self.object_y, self.object_x)
            # Reducimos de 0.8 a 0.6 para no chocar con el suelo
            self.send_arm_command([j1, 0.65, 0.35, 0.3]) 
            self.state = 'LOWERING'
            
        elif self.state == 'LOWERING':
            # Cierra gripper
            self.get_logger().info('Cerrando gripper...')
            self.send_gripper_command(-0.01) # Cerrar
            self.state = 'GRASPING'
            
        elif self.state == 'GRASPING':
            # Levanta
            self.get_logger().info('Levantando objeto...')
            j1 = np.arctan2(self.object_y, self.object_x)
            self.send_arm_command([j1, -0.5, 0.0, 0.5])
            self.state = 'LIFTING'
            
        elif self.state == 'LIFTING':
            self.get_logger().info('Ciclo completado.')
            self.state = 'IDLE'

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()