## 1 环境配置
requirments.txt为环境配置
安装方式为
```commandline
pip install -r requirements.txt
```

## 2 测试
cd 至codes目录下，直接运行
```commandline
python test.py
```

## 3 训练
cd 至codes目录下，直接运行
```commandline
python train.py
```

## 4 配置文件
配置文件存放与./codes/config.yaml

训练和测试前，需要准备好数据，准备的方式可以参考
“./datasets/TrainTest_2020/”。

我自己写了一个预处理，若需调整，可参照"./codes/TifPreprocess.py"进行数据准备。