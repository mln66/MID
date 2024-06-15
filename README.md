# Meta Invariance Defense Towards Generalizable Robustness to Unknown Adversarial Attacks

## Describe
This is the official project page of the paper "Meta Invariance Defense Towards Generalizable Robustness to Unknown Adversarial Attacks" TPAMI 2024.
(https://arxiv.org/abs/2404.03340)
## Running enviroment
- pytorch 0.4.1
- python 3.6
- advertorch 
- cuda 8.0

## Program Running
1. The normal train module in Training_MNIST.py is used to train a baseline model (target model).
2. The teacher model training module in Training-MNIST.py is used to train the teacher module, including an encoder, a decoder, and a classifier.
3. The main runtime module of MID is meta-depense-MNIST_cls_simi_cyc.py, which needs to be run with the teacher model and target model ready.

## Model weights
We have provided model weights on the mnist dataset, which are saved in the saving models folder. You can directly load these models for testing.
1. The weight of the target model is ./revision/MNIST_LeNet5.pkl.
2. The weight of the teacher's network is LeNet5_MINIST_AE.pkl. 
3. The weight of the MID training model is ./final_models/encoder_adv_addclean_MNIST_cls_simi_cyc217.pkl.  
If you want to run other datasets, you can modify the dataset loading based on the code.

## Citation
If you find this code useful in your research then please cite
~~~
@article{zhang2024meta,
  title={Meta Invariance Defense Towards Generalizable Robustness to Unknown Adversarial Attacks},
  author={Zhang, Lei and Zhou, Yuhang and Yang, Yi and Gao, Xinbo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
~~~
## Contact
If you have any problem about our code, feel free to contact us.
* Yuhang Zhou (23B951015@stu.hit.edu.cn)  
* Lei Zhang (leizhang@cqu.edu.cn)