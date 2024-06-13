# Meta Invariance Defense
## 运行环境
- pytorch 0.4.1
- python 3.6
- advertorch 
- cuda 8.0

    

## 程序运行
1. Training_MNIST.py中的normal train模块用于训练一个baseline模型（目标模型）
2. Training_MNIST.py中的教师模型训练模块用于训练教师模块，包括一个编码器，一个解码器和一个分类器
3. meta_defense_MNIST_cls_simi_cyc.py为MID的主要运行模块，需要在准备好教师模型和目标模型的前提下运行。


# MNIST运行结果

dd&|姓名   | 年龄 |  工作 |
| :----- | :--: | -------: |
| 小可爱 |  18  | 吃可爱多 |
| 小小勇敢 |  20  | 爬棵勇敢树 |
| 小小小机智 |  22  | 看一本机智书 |&

$$\(\sqrt{3x-1}+(1+x)^2\)$$  

