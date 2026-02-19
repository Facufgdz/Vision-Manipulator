#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')
        self.subscription_img = self.create_subscription(Image, '/static_camera/image_raw', self.image_callback, 10)
        self.subscription_info = self.create_subscription(CameraInfo, '/static_camera/camera_info', self.info_callback, 10)
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.get_logger().info('Buscando el cubo verde para calibrar...')

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f'Matriz de cámara recibida: {self.camera_matrix}')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Máscara para el verde (cubo)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Seleccionar el contorno más grande
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Publicar coordenadas de pixel
                self.get_logger().info(f'Cubo detectado en Pixel: ({cx}, {cy})')
                self.get_logger().info('Calibración sugerida: Pixel ({}, {}) -> Robot (0.18, 0.0)'.format(cx, cy))
                
                # Dibujar para visualización (aunque no lo veamos directamente)
                cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)
                
        # Guardar una imagen de prueba para verificar
        cv2.imwrite('/home/facufgdz/ros2_ws/calibration_view.jpg', cv_image)

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
