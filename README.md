Eye Gaze Detection System
Overview
This project aims to develop a resource-efficient eye gaze detection system using MobileNetV3 as the baseline model. Although MobileNetV3 is already lightweight, we aim to further reduce its resource complexity through pruning techniques.

Structure
Baseline Model: We utilize the MobileNetV3 model as our starting point due to its efficiency and effectiveness for mobile and edge devices.

Pruning Method:

Magnitude Pruning: We employ a magnitude pruning algorithm based on the L2 norm pruning criteria. This method helps in reducing the number of parameters in the model, thereby decreasing its resource requirements while maintaining performance.
Directory Layout:

magnitude: Contains the implementation of the magnitude pruning algorithm.
pruned: Houses the models that have been pruned using the magnitude pruning technique.
finetuned: Contains the models that have been finetuned after pruning to regain any lost accuracy.
Usage
Pruning:

Navigate to the magnitude_pruning folder to access and run the pruning scripts in Magnitudepruner.py.
Pruned models will be saved in the pruned folder.
Finetuning:

After pruning, models need to be finetuned to ensure they perform well on the eye gaze detection task.
Finetuning scripts are also available, and the resultant finetuned models are stored in the finetuned folder.
Conclusion
By combining MobileNetV3 with L2 norm-based magnitude pruning, we aim to create an efficient eye gaze detection system that is suitable for deployment on resource-constrained devices. The pruned and finetuned models strike a balance between performance and computational efficiency.
