# Vision-Manipulator 🦾👁️

Este repositorio contiene una implementación personalizada del **OpenMANIPULATOR-X** en **ROS 2 Humble**, integrada con un sistema de visión artificial para realizar tareas de Pick and Place automáticas en **Gazebo**.

## 🚀 Características Implementadas

1.  **Visión Artificial con OpenCV**: Nodo de procesamiento de imágenes que detecta un objeto verde mediante filtrado de color (HSV).
2.  **Cámara Cenital Estática**: Integración de una cámara en la simulación posicionada a 1 metro de altura, alineada perfectamente para una vista de planta del área de trabajo.
3.  **Objeto Graspable**: Creación de un cubo verde de 3cm con propiedades físicas (masa, inercia, fricción) optimizadas para ser agarrado por el gripper.
4.  **Máquina de Estados de Control**: Lógica automática que gestiona la secuencia:
    *   **Detección**: Identifica el centroide del objeto.
    *   **Aproximación**: Mueve el gripper sobre el objeto.
    *   **Descenso**: Baja la pinza a nivel del suelo con precisión.
    *   **Agarre**: Acciona el gripper.
    *   **Elevación**: Levanta el objeto para completar el ciclo.

---

## 🛠️ Instalación y Compilación

Asegúrate de tener instalado ROS 2 Humble y las dependencias correspondientes (Gazebo y OpenCV).

```bash
# Navegar al workspace
cd ~/ros2_ws

# Clonar el repositorio (si no lo tienes aún)
git clone https://github.com/Facufgdz/Vision-Manipulator.git src/open_manipulator

# Instalar dependencias necesarias
sudo apt update
sudo apt install ros-humble-cv-bridge ros-humble-gazebo-ros-pkgs ros-humble-control-msgs

# Compilar el proyecto
colcon build --packages-select open_manipulator open_manipulator_x_bringup
source install/setup.bash
```

---

## 🏃 Ejecución

### 1. Iniciar la Simulación (Gazebo)
Lanza el entorno con el brazo, la cámara y el cubo:
```bash
ros2 launch open_manipulator_x_bringup gazebo.launch.py
```

### 2. Iniciar el Nodo de Visión y Control
En una terminal nueva, ejecuta el script de seguimiento:
```bash
source install/setup.bash
ros2 run open_manipulator follow_camera.py
```
*Se abrirán dos ventanas de OpenCV mostrando la cámara original y la máscara de detección.*

---

## 🎮 Comandos Útiles

### Mover la cámara dinámicamente
Si deseas ajustar la posición de la cámara mientras Gazebo está abierto, puedes usar el servicio de estados:
```bash
ros2 service call /gazebo/set_entity_state gazebo_msgs/srv/SetEntityState "{state: {name: 'static_camera', pose: {position: {x: 0.4, y: 0.0, z: 1.2}, orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}}}}"
```

### Limpiar procesos de Gazebo
Si la simulación se queda trabada, usa este comando para cerrar todo:
```bash
pkill -9 gzserver; pkill -9 gzclient; pkill -9 -f gazebo
```

---

## 📁 Estructura del Proyecto
*   `open_manipulator/follow_camera.py`: Nodo principal de Python (Visión + Control).
*   `models/graspable_cube/`: Definición SDF del cubo verde.
*   `open_manipulator_x_bringup/launch/gazebo.launch.py`: Configuración de lanzamiento y spawneo.
*   `open_manipulator_x_bringup/worlds/empty_world.model`: Mundo con plugin de estados habilitado.

---
**Desarrollado para:** Facundo FG (Vision-Manipulator Project)
