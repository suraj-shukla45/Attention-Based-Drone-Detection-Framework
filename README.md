# Real-time-drone-detection
This project centers on real-time drone detection and monitoring within surveillance settings by deep learning and computer vision techniques. It aims to reliably identify small, distant UAVs under challenging conditions such as fog, low light, cluttered backgrounds, and long-range views. To enhance detection accuracy, the YOLO11n model was improved with attention mechanisms like Coordinate Attention (CA), GELAN-E feature enhancement methods, and synthetic image augmentation to diversify the dataset and strengthen model robustness.
## Dataset Description 
## Detfly:
The model was trained and evaluated using the Det-Fly dataset, a UAV detection dataset designed for air-to-air visual drone detection using monocular cameras. The dataset contains more than 13,000 images captured by a flying DJI Mavic2 UAV and includes various real-world surveillance scenarios.
Det-Fly covers multiple environmental conditions including sky, urban, field, and mountain backgrounds, along with different viewing angles such as front, top, and bottom views. A significant portion of the dataset contains very small drone targets, making it suitable for tiny-object detection research. 
To further improve model robustness and generalization, synthetic drone images and augmentation techniques were also incorporated into the training dataset for handling complex environments, long-range detection, and challenging lighting conditions.
## Synthetic_Data:
To enhance environmental diversity and boost model robustness, synthetic UAV images were created using Stable Diffusion, which is based on the Latent Diffusion Model (LDM). The generated images simulated various weather conditions, including fog, haze, low light, rain, and other complex atmospheric scenarios. This augmentation strategy improved the model's capability to detect drones in difficult real-world surveillance environments, ultimately enhancing generalization, feature learning, and detection accuracy for small and distant UAV targets.
## Base Architecture (YOLO11n)

The initial model architecture was based on YOLO11n, a lightweight real-time object detection framework optimized for fast inference and efficient multi-scale feature extraction. YOLO11n was selected due to its low computational cost and strong performance in detecting small aerial targets in surveillance environments.
<img width="1536" height="1024" alt="ChatGPT Image May 10, 2026, 06_52_35 PM" src="https://github.com/user-attachments/assets/cf35aebf-816e-4b30-8cdf-b3f99ac01be3" />



## Enhanced Architecture (YOLO11n + GELAN-E + Coordinate Attention)

To improve drone detection performance, the baseline YOLO11n architecture was enhanced by integrating GELAN-E blocks and Coordinate Attention (CA) modules. GELAN-E was incorporated to strengthen feature reuse and gradient flow, while Coordinate Attention improved spatial and channel-aware feature representation for better localization of small and distant UAVs. These enhancements helped improve robustness, feature extraction capability, and detection accuracy in complex real-world environments such as fog, low-light, cluttered backgrounds, and long-range surveillance scenes.
<img width="1536" height="1024" alt="with  attentions" src="https://github.com/user-attachments/assets/d8bfd6f9-460f-475e-bf09-f821709be527" />
