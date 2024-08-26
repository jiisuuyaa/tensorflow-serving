import requests
import json
import numpy as np
from tensorflow.keras.preprocessing import image


# 이미지 불러오기 및 전처리
img_path = 'C:/Users/jiisuu/Desktop/dog.jpg'  # 이미지 파일 경로
img = image.load_img(img_path, target_size=(64, 64))  # 모델의 입력 크기에 맞게 조정
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # 배치 차원을 추가
img_array = img_array.astype('float32') / 255.0  # 정규화 수행


# 입력 데이터를 리스트 형식으로 변환
input_data = img_array.tolist()


# TensorFlow Serving 서버 URL 설정
url = 'http://localhost:8501/v1/models/tensor_cnn:predict'


# 요청 본문 및 헤더 설정
data = json.dumps({"instances": input_data})
headers = {"content-type": "application/json"}


# 데이터 전송 확인
print("Data sent:", data)


# POST 요청 보내기
response = requests.post(url, data=data, headers=headers)
predictions = response.json()


# 예측 결과 출력
print(predictions)


predicted_value = predictions['predictions'][0][0]


if predicted_value > 0.5:
    print("'dog'")
else:
    print("'cat'")
