# Attention-Based-Drone-Detection-Framework
This project focuses on real-time drone detection and monitoring in surveillance environments using deep learning and computer vision techniques. It aims to reliably detect small, distant UAVs under challenging conditions such as fog, low light, cluttered backgrounds, and long-range scenes. To improve detection accuracy, the YOLO11n model was enhanced with attention mechanisms, including Coordinate Attention (CA).
## Base Architecture (YOLO11n)
The initial model architecture was based on YOLO11n, a lightweight real-time object detection framework optimized for fast inference and efficient multi-scale feature extraction. YOLO11n was selected due to its low computational cost and strong performance in detecting small aerial targets in surveillance environments.
<img width="800" height="512" alt="ChatGPT Image May 10, 2026, 06_52_35 PM" src="https://github.com/user-attachments/assets/cf35aebf-816e-4b30-8cdf-b3f99ac01be3" />
## Enhanced Architecture (YOLO11n Coordinate Attention)
To improve drone detection performance, the baseline YOLO11n architecture was enhanced by integrating Coordinate Attention (CA) modules. Coordinate Attention improved spatial and channel-aware feature representation, enabling better localization of small and distant UAVs. These enhancements boosted the model's robustness, feature extraction capability, and detection accuracy in complex real-world environments, including fog, low-light conditions, cluttered backgrounds, and long-range surveillance scenes.
<img width="800" height="512" alt="2222" src="https://github.com/user-attachments/assets/1b295067-3c78-4fa2-b055-bb6083cb19b3" />

## Dataset Description 
## Detfly:
The model was trained and evaluated using the Det-Fly dataset, a UAV detection dataset designed for air-to-air visual drone detection using monocular cameras. The dataset contains more than 13,000 images captured by a flying DJI Mavic2 UAV and includes various real-world surveillance scenarios.
Det-Fly covers multiple environmental conditions including sky, urban, field, and mountain backgrounds, along with different viewing angles such as front, top, and bottom views. A significant portion of the dataset contains very small drone targets, making it suitable for tiny-object detection research. 
To further improve model robustness and generalization, synthetic drone images and augmentation techniques were also incorporated into the training dataset for handling complex environments, long-range detection, and challenging lighting conditions.
