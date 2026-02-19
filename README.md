# Vision-Manipulator 🦾👁️

Este repositorio contiene una implementación personalizada del **OpenMANIPULATOR-X** en **ROS 2 Humble**, integrada con un sistema de visión artificial para realizar tareas de Pick and Place automáticas en **Gazebo**.

## 🚀 Características Implementadas

1.  **Visión Universal (No-Color Dependiente)**: Nodo de procesamiento de imágenes que detecta cualquier objeto que resalte sobre el suelo mediante umbralización de grises y máscaras ROI (Region of Interest) para evitar la auto-detección del propio brazo.
2.  **Calibración Maestra Óptica**: Implementación de una matriz de transformación de píxeles a coordenadas de robot basada en una pose de calibración fija (Master Pose).
3.  **Física de Agarre Avanzada**: Ajuste fino de parámetros de contacto en Gazebo (`kp`, `kd`, `mu`, `minDepth`) tanto en el Gripper como en los objetos para evitar que los objetos salgan disparados o se resbalen.
4.  **Torque Potenciado**: Modificación del URDF para incrementar el límite de esfuerzo (`effort`) de los eslabones (de 1.0 a 10.0), mejorando la respuesta física ante cargas en la simulación.
5.  **Control de Trayectorias en Tiempo Real**: Lógica de seguimiento continuo (10Hz) que ajusta dinámicamente la extensión del brazo basándose en el feedback visual.

---

## ✅ Avances y Soluciones
*   **Agarre Estable**: Se solucionó el problema de colisiones violentas reduciendo la velocidad de aproximación y ajustando la amortiguación de los contactos.
*   **Levantamiento Exitoso**: El robot es capaz de localizar el cubo, descender sin golpear el suelo (gracias a la calibración de altura Z) y elevarlo de forma consistente.
*   **Independencia de Color**: El sistema ahora rastrea cualquier objeto (cajas, esferas, cilindros) que entre en su campo de visión.

## ⚠️ Problemas Pendientes (Work in Progress)
*   **Rotación de Base (Eje Y)**: A pesar de implementar cálculos trigonométricos (`atan2`) y aumentar el torque, la rotación de la base (`joint1`) presenta una respuesta inconsistente en Gazebo al intentar seguir movimientos laterales agresivos.
*   **Suavizado de Trayectorias**: El control a alta frecuencia (10Hz) puede generar vibraciones menores durante el seguimiento dinámico que necesitan ser filtradas.

---

## 🛠️ Instalación y Compilación

Asegúrate de tener instalado ROS 2 Humble y las dependencias correspondientes (Gazebo y OpenCV).

```bash
# Navegar al workspace
cd ~/ros2_ws

# Instalar dependencias necesarias
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-gazebo-ros-pkgs ros-humble-control-msgs

# Compilar el proyecto completo incluyendo descripciones
colcon build --symlink-install
source install/setup.bash
```

---

## 🏃 Ejecución

### 1. Iniciar la Simulación (Gazebo)
```bash
ros2 launch open_manipulator_x_bringup gazebo.launch.py
```

### 2. Iniciar el Seguimiento por Visión
```bash
# Seguir cualquier objeto detectado
ros2 run open_manipulator follow_with_camera.py
```

### 3. (Opcional) Calibración de Cámara
Si cambias la altura de la cámara, puedes recalibrar los píxeles usando:
```bash
ros2 run open_manipulator camera_calibration.py
```

---

## 🎮 Comandos Útiles

### Limpiar procesos de Gazebo
```bash
pkill -9 gzserver; pkill -9 gzclient; pkill -9 -f gazebo
```

---

## 📁 Estructura del Proyecto
*   `open_manipulator/follow_with_camera.py`: Nodo de Seguimiento Dinámico (Visión Universal + PID).
*   `open_manipulator/camera_calibration.py`: Herramienta de mapeo Pixel-to-Robot.
*   `open_manipulator_x_description/urdf/`: URDF con límites de torque potenciados.
*   `models/graspable_cube/`: Modelo optimizado con alta fricción.

---
**Desarrollado para:** Facundo FG (Vision-Manipulator Project)
