# Agricultural field segmentation  
This repository contains code for deep learning project. 

All training and models parameters are in config files. For our models we used following configs: `exp_dl_hrnet.yaml` for hrnet and `exp_dl_unet.yaml` for unet resnet architecture. training is run by command

```
python3 train.py -c configs/exp_dl_hrnet.yaml
```

for configs we used `configargparse` so this configs could be easily changed



For running trained model   


```
python3 train.py -c logs/{name of your run}/config.yaml
```

All training data should be loaded into workdir directory.