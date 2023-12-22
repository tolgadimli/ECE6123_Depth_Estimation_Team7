# ECE6123 Depth Estimation from Monocular Images - Team 7

Depth estimation is one of the most essential tasks in computer vision and has considerable importance in many applications ranging from autonomous driving to augmented reality. Extracting such information accurately from a single image still remains an open challenge in this field. In this project, we first review the literature for depth estimation from monocular image (DEFMI) while mainly focusing on the deep learning (DL) based approaches in the supervised learning (SL) scheme. These approaches make architectural changes to the network and/or propose a new objective function to be optimized. Although some approaches that introduce an auxiliary task exist in the literature, we do not include a competitor from this type in our evaluation since it requires additional dense labeling of the depth dataset and not all the DEFMI datasets have the extra labels required for the auxiliary task. Overall, we compare the performances of a baseline method with four different methods from the literature that bring a considerable novelty rather than a small improvement or a tweak to improve a well-established existing method. Our evaluation of the selected methods is further enriched with the interpretation of their drawbacks/strengths. Although the main goal of this project is to evaluate and compare the performances of different DEFMI techniques, it was initially also desired to propose a novel method or improvement for this task by using the insights and potential caveats obtained from the experiments that we conduct. On the other hand, our literature search showed that our idea has already been explored. For our experiments, we use the most commonly preferred datasets for DEFMI tasks by the community such as NYU Depth V2, SUN RGB-D and DIODE. For our quantitative comparison, we are reporting the final Root Mean Square error (RMSE), Relative Mean Absolute Error (Rel. MAE), Silog, and Threshold Accuracy scores since these metrics are adopted by the community to ensure a systematic and comprehensive assessment. For all of our implementations, we use Python language and the PyTorch library as the deep learning framework. 

- MYUv2 dataset is used from huggingface: https://huggingface.co/datasets/sayakpaul/nyu_depth_v2 \
- SUN RGB-D dataset can be downloaded from here: https://rgbd.cs.princeton.edu/ \
- The DIODE dataset can be downloaded from here: https://diode-dataset.org/ \

The datasets should be located in `../datasets`.
