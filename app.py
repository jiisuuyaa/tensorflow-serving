import streamlit as st
import requests
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# Streamlit 애플리케이션 설정
st.set_page_config(
    page_title="강아지 vs 고양이 분류기",
    page_icon="🐶🐱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title('🐾 Tensorflow 이미지 분류 모델 SERVING 과정 함께 보시져! 🐶🐺')
st.markdown("**강아지/고양이** 사진을 업로드해 주세요. \n서빙된 모델이 이미지를 분류해 드릴게요!")

# 사이드바에 정보 추가
with st.sidebar:
    st.markdown("## model serving")
    st.markdown("이 앱은 TensorFlow Serving을 통해 \n서빙된 TensorFlow 모델을 사용하여 \n강아지와 고양이 이미지를 분류합니다.")
    st.markdown("이미지를 업로드하면 \n실시간 api가 연동되며 \n그 이미지가 강아지인지 고양이인지 알려줍니다. 🐶🐱")

# 이미지 업로드 위젯
uploaded_file = st.file_uploader("이미지를 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지를 PIL 포맷으로 로드
    img = image.load_img(uploaded_file, target_size=(64, 64))
    
    # 이미지를 배열로 변환하고 전처리
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array.astype('float32') / 255.0  # 정규화 수행

    # 입력 데이터를 리스트 형식으로 변환
    input_data = img_array.tolist()

    # TensorFlow Serving 서버 URL 설정
    url = 'http://localhost:8501/v1/models/tensor_cnn:predict'

    # 요청 본문 및 헤더 설정
    data = json.dumps({"instances": input_data})
    headers = {"content-type": "application/json"}

    # 로딩 애니메이션 추가
    with st.spinner('TensorFlow Serving을 통해 이미지 분류 진행 중 ... 잠시만 기다려 주세요!'):
        time.sleep(2)  # 로딩 시간을 추가하여 사용자가 대기 중임을 인식할 수 있도록 함
        # POST 요청 보내기
        response = requests.post(url, data=data, headers=headers)
        predictions = response.json()

        # 예측 결과 출력
        predicted_value = predictions['predictions'][0][0]

        if predicted_value > 0.5:
            st.success("결과: **강아지** 🐶")
        else:
            st.success("결과: **고양이** 🐱")

        # 이미지를 화면에 표시
        st.image(uploaded_file, caption='업로드한 이미지입니다.', use_column_width=True)

    # 애플리케이션 최종 화면에 감사 인사 추가
    st.markdown("### 감사합니다! 🥰")
    st.markdown("모델링 코드를 짜지 않아도 모델 서빙을 통해 서비스를 이용할 수 있습니다 ~!")

else:
    st.info("먼저 이미지를 업로드해 주세요!")
