# Note to run MiDas on 10.27.32.185

Directory: ``` /media/giang/VulPatchClassifier```

Docker command: ```docker run --name giang_dl -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -it --rm  --shm-size 16G --gpus all -v /media/giang/:/giang/ giang_deep_learning```

## VulFixMiner
Predicted probabilities Java: ```probs/huawei_pred_prob_java.csv```

Predicted probabilities Python: ```probs/huawei_pred_prob_python.csv```

## LApredict
To train and evaluate LApredict: ```python la_classifier.py```

Predicted probabilities Java: ```probs/la_prob_java.txt```

Predicted probabilities Python: ```probs/la_prob_python.txt```

## LOC-sensitive model
Predicted probabilities Java: ```probs/dummy_sorting_java_prob.txt```

Predicted probabilities Python: ```probs/dummy_sorting_python_prob.txt```
