# 用ImageAI识别图像
# http://developer.51cto.com/art/201804/571353.htm

from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd()
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
# 加载模型文件(用ResNet模型，对ImageNet-1000数据集训练后的keras模型文件)
prediction.setModelPath( execution_path + "/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()
# 预测
predictions, percentage_probabilities = prediction.predictImage(execution_path + "\samples\sample.jpg", result_count=5)
# 输出预测结果：分类, 可能性(%)
for index in range(len(predictions)):
    print(predictions[index] + " : " + str(percentage_probabilities[index]))
