'''
加载训练好的模型，再对图片进行物体检测
'''

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel()

input_image = os.path.join(execution_path, 'data/image.png')
output_image = os.path.join(execution_path, 'data/imagenew.png')
detections = detector.detectObjectsFromImage(input_image=input_image
                                        , output_image_path=output_image
                                        , extract_detected_objects=False)

print('object : probability')
for object in detections:
    print(object['name'] + ' : ' + str(object['percentage_probability']))
