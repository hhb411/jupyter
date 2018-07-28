from train import main
from predict import predict
import os
print(os.getcwd())
"""
traffic-sign
"""
# main('../data/traffic-sign/train', '../data/traffic-sign/test', 'traffic_sign2.model', 'traffic_sign2.png')

# main('../data/traffic-sign-small/train', '../data/traffic-sign-small/test', 'traffic_sign_small.model', 'traffic_sign_small.png')
# predict('traffic_sign_small.model', '../data/predict/01565_00002.png', '')

"""
weather-sign
"""
main('../data/weather-sign/train', '../data/weather-sign/test', 'weather_sign.model', 'weather_sign.png', True)

# predict('weather_sign.model', '../data/weather-sign/predict/00000/0.png', ' ')
# predict('weather_sign.model', '../data/weather-sign/predict/00000/1.png', ' ')
# predict('weather_sign.model', '../data/weather-sign/predict/00000/2.png', ' ')
#
# predict('weather_sign.model', '../data/weather-sign/predict/00001/0.png', ' ')
# predict('weather_sign.model', '../data/weather-sign/predict/00001/1.png', ' ')

# predict('weather_sign.model', '../data/weather-sign/predict/00002/0.png', ' ')
# predict('weather_sign.model', '../data/weather-sign/predict/00002/1.png', ' ')
# predict('weather_sign.model', '../data/weather-sign/predict/00002/2.png', ' ')
# predict('weather_sign.model', '../data/weather-sign/predict/00002/3.png', ' ')

"""

"""