from sys import displayhook
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import pandas as pd
import joblib
import yaml
import time
import openai
import backoff
from openai import OpenAIError, RateLimitError
import os

# 절대 경로 설정
base_path_data = "C:/Users/vivid/Desktop/contest/Final/Data"
base_path_model = "C:/Users/vivid/Desktop/contest/Final/Final_Model"
original_data_path = "C:/Users/vivid/Desktop/contest/Data/original"
config_path = "C:/Users/vivid/Desktop/contest/Config/config.yaml"

# 데이터 로드
industry = pd.read_excel(os.path.join(base_path_data, 'korean_industry.xlsx'), dtype={'코드': str})
SMEs = pd.read_csv(os.path.join(original_data_path, 'Regional_specialization_SMEs.csv'), encoding='cp949')
Factories = pd.read_csv(os.path.join(base_path_data, 'Factory_total.csv'))
Factories = Factories[Factories['상태'] == '진행']  # 현재 진행 중인 데이터만 가져오기

local_governments = {
    '1': '서울특별시', 
    '2': '부산광역시', 
    '3': '대구광역시', 
    '4': '인천광역시', 
    '5': '광주광역시', 
    '6': '대전광역시', 
    '7': '울산광역시', 
    '8': '세종특별자치시', 
    '9': '경기도', 
    '10': '강원특별자치도', 
    '11': '충청북도', 
    '12': '충청남도', 
    '13': '전북특별자치도', 
    '14': '전라남도', 
    '15': '경상북도', 
    '16': '경상남도', 
    '17': '제주특별자치도'
}

local_governments_english = {
    '서울특별시': 'Seoul',
    '부산광역시': 'Busan',
    '대구광역시': 'Daegu',
    '인천광역시': 'Incheon',
    '광주광역시': 'Gwangju',
    '대전광역시': 'Daejeon',
    '울산광역시': 'Ulsan',
    '세종특별자치시': 'Sejong',
    '경기도': 'Gyeonggi',
    '강원특별자치도': 'Gangwon',
    '충청북도': 'Chungbuk',
    '충청남도': 'Chungnam',
    '전북특별자치도': 'Jeonbuk',
    '전라남도': 'Jeonnam',
    '경상북도': 'Gyeongbuk',
    '경상남도': 'Gyeongnam',
    '제주특별자치도': 'Jeju'
}

# local_goverment 선택
print("공장 부지 선정을 위한 지자체를 선택해주세요.")
for key, value in local_governments.items():
    print(f"{key}. {value}")

user_input_local_goverment = input("지자체를 선택해주세요: ")
if user_input_local_goverment in local_governments:
    korean_name = local_governments[user_input_local_goverment]
    english_name = local_governments_english[korean_name]
    print(f'선택한 지자체는 "{korean_name}" 입니다\n적합한 모델을 불러오는 중입니다. 잠시만 기다려주세요.')
    
    # 데이터 및 모델 로드
    data_path = os.path.join(base_path_data, f'{english_name}_encoding_data.csv')
    data = pd.read_csv(data_path)
    lgbm_best_model = joblib.load(os.path.join(base_path_model, f'{english_name}_best_model.pkl'))
    label_encoders = joblib.load(os.path.join(base_path_model, f'{english_name}_label_encoders_lgbm.pkl'))
    original_values = joblib.load(os.path.join(base_path_model, f'{english_name}_original_values_lgbm.pkl'))

    # 대표업종 라벨 인코더 로드
    label_encoder_industry = label_encoders['대표업종']

    # 데이터프레임의 최빈값 계산
    most_frequent_values = data.mode().iloc[0].astype(int)

    print("모델 및 인코딩 준비 완료")
else:
    raise ValueError("Invalid input for local_goverment. Please enter a number from the list.")

# 3. 입력받지 못한 값, 대표 업종 코드에 대한 최빈값으로 결측치를 채움
mode_values_by_business = data.groupby('대표업종').agg(lambda x: x.mode().iloc[0])
most_frequent_representative_business = data['대표업종'].mode().iloc[0] #대표업종 최빈값
most_frequent_representative_business_encoder = str(label_encoder_industry.inverse_transform([most_frequent_representative_business])[0])

# 사용자 입력 데이터 받기
user_input_보유구분 = input("어떤 공장을 보유하고 싶으신가요? (임대: 1 / 자가: 2): ")
if user_input_보유구분 == '1':
    user_input_보유구분 = '임대'
elif user_input_보유구분 == '2':
    user_input_보유구분 = '자가'
else:
    raise ValueError("적당하지 않은 값 입니다. 1 / 2를 눌러주세요")

user_input_등록구분 = input("공장 등록구분을 선택해주세요.(1: 등록변경 / 2: 부분등록 / 3: 신규등록 / 4: 완료신고 )")
등록구분_dict = {'1': '등록변경', '2': '부분등록', '3': '신규등록', '4': '완료신고'}
if user_input_등록구분 in 등록구분_dict:
    user_input_등록구분 = 등록구분_dict[user_input_등록구분]
else:
    raise ValueError("적당하지 않은 값 입니다. 1 / 2 / 3 / 4 를 눌러주세요")

user_input_공장규모 = input("공장 규모를 선택해주세요.(1: 소기업 / 2: 중기업 / 3: 대기업 )")
공장규모_dict = {'1': '소기업', '2': '중기업', '3': '대기업'}
if user_input_공장규모 in 공장규모_dict:
    user_input_공장규모 = 공장규모_dict[user_input_공장규모]
else:
    raise ValueError("적당하지 않은 값 입니다. 1 / 2 / 3/ 를 눌러주세요")

user_input_남자종업원 = int(input("남자종업원 수를 입력해주세요: "))
user_input_여자종업원 = int(input("여자종업원 수를 입력해주세요: "))
user_input_외국인남자종업원 = int(input("외국인 남자종업원 수를 입력해주세요: "))
user_input_외국인여자종업원 = int(input("외국인 여자종업원 수를 입력해주세요: "))

# 종업원합계 자동 계산
user_input_종업원합계 = user_input_남자종업원 + user_input_여자종업원 + user_input_외국인남자종업원 + user_input_외국인여자종업원

user_input_대표업종 = int(input("대표업종 코드를 입력해주세요. (예: 18111): "))

# 최빈값 계산
most_frequent_representative_business_encoder = data['대표업종'].mode()[0]

# 대표업종 코드 검증 및 대체
def get_industry_name_or_default(user_input_code, industry_df, label_encoder, default_code):
    user_input_code_str = str(user_input_code)
    if user_input_code_str in industry_df['코드'].values:
        entry_value = industry_df.loc[industry_df['코드'] == user_input_code_str, '항목명'].values[0]
        return user_input_code_str, entry_value
    else:
        print(f"입력한 대표업종 코드({user_input_code})가 데이터에 존재하지 않습니다.")
        default_code_str = str(default_code)
        default_name = industry_df.loc[industry_df['코드'] == default_code_str, '항목명'].values[0]
        print(f"대표업종 코드를 최빈값인 {default_code_str} ({default_name})으로 대체합니다.")
        return default_code_str, default_name

# 최빈값 가져오기
most_frequent_representative_business = data['대표업종'].mode().iloc[0]
most_frequent_representative_business_code = str(label_encoder_industry.inverse_transform([most_frequent_representative_business])[0])

# 사용자 입력 코드 검증 및 대체
final_대표업종_code, final_대표업종_name = get_industry_name_or_default(user_input_대표업종, industry, label_encoder_industry, most_frequent_representative_business_code)

# 사용자 입력 데이터
user_input = {
    '보유구분': user_input_보유구분, 
    '등록구분': user_input_등록구분, 
    '공장규모': user_input_공장규모, 
    '남자종업원': user_input_남자종업원, 
    '여자종업원': user_input_여자종업원, 
    '외국인남자종업원': user_input_외국인남자종업원, 
    '외국인여자종업원': user_input_외국인여자종업원, 
    '종업원합계': user_input_종업원합계,
    '대표업종': final_대표업종_code
}

encoded_user_input = {}

for col, value in user_input.items():
    if col == '대표업종':
        # 대표업종에 대해 라벨 인코딩 적용
        try:
            encoded_user_input[col] = label_encoder_industry.transform([value])[0]
        except ValueError:
            print(f"대표업종 코드({value})가 라벨 인코더에 존재하지 않습니다. 최빈값으로 대체합니다.")
            encoded_user_input[col] = label_encoder_industry.transform([most_frequent_representative_business_code])[0]
    elif col in label_encoders:
        # 해당 컬럼에 대해 라벨 인코딩 적용
        encoded_user_input[col] = label_encoders[col].transform([value])[0]
    else:
        encoded_user_input[col] = value

# 결측치 채우기 (최빈값으로)
for col in data.columns:
    if col not in encoded_user_input:
        if col in most_frequent_values.index:
            encoded_user_input[col] = most_frequent_values[col]

# DataFrame 형태로 변환
input_df = pd.DataFrame([encoded_user_input])

# '구' 열 제거
if '구' in input_df.columns:
    input_df = input_df.drop(columns=['구'])

model_features = lgbm_best_model.booster_.feature_name()

encoded_user_input = {}

# 입력 데이터에 모든 피처를 포함하도록 수정
for col in model_features:
    if col not in encoded_user_input:
        encoded_user_input[col] = most_frequent_values[col] if col in most_frequent_values.index else 0

input_df = pd.DataFrame([encoded_user_input])
input_df = input_df[model_features]

input_df = pd.DataFrame([encoded_user_input])

# 모델이 학습할 때 사용한 열 순서로 정렬
model_features = ['공장구분', '설립구분', '입주형태', '보유구분', '등록구분', '남자종업원', '여자종업원', 
                  '외국인남자종업원', '외국인여자종업원', '종업원합계', '공장규모', '용도지역', '지목', 
                  '용지면적', '제조시설면적', '부대시설면적', '건축면적', '지식산업센터명', '대표업종']

input_df = input_df[model_features]

# 인코딩된 값을 원래 값으로 변환
decoded_user_input = {}
for col in input_df.columns:
    if col in label_encoders:
        decoded_user_input[col] = label_encoders[col].inverse_transform(input_df[col])[0]
    else:
        decoded_user_input[col] = input_df[col].values[0]

# 결과 출력
decoded_df = pd.DataFrame([decoded_user_input])

# 4. 모델 추론 수행
prediction = lgbm_best_model.predict(input_df)

# 결과를 원래 값으로 변환
predicted_label = prediction[0]
predicted_class = original_values['구'][predicted_label]
print(f"모델 예측 결과: {predicted_class}")
print()

# 주력산업 찾기
user_input_local_goverment = int(user_input_local_goverment)
selected_industry = SMEs[SMEs['연번'] == user_input_local_goverment]['주력산업'].values[0]

# 결과 출력
print(f"선택한 지자체의 주력산업은: {selected_industry}. \n\n산업에 대한 장점, 단점, 전망에 대해서 분석중입니다. \n잠시만 기다려주세요.")
print()
# ## OPENAI 통해서 산업 장단점, 전망 파악하기

#yaml 파일 열기
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

Service_key = config['ServiceKey']['OpenAIkEY']
openai.api_key = Service_key

from openai import OpenAI, OpenAIError, RateLimitError
import time
import backoff

# OpenAI 클라이언트 초기화
client = OpenAI(
    api_key = Service_key 
)

# 재시도 로직을 사용한 함수
@backoff.on_exception(backoff.expo, (OpenAIError, RateLimitError), max_tries=5)
def get_completion(**kwargs):
    return client.chat.completions.create(**kwargs)

message_content = f"{final_대표업종_name}에 관련해서 장점, 단점, 전망, 결론에 대해서 사업을 시작하기 위한 사람들을 위해 자세하게 설명해줘."

# 요청 데이터 설정
data = {
    'model': 'gpt-4o',  # 최신 모델 사용
    'messages': [{'role': 'user', 'content': message_content}],
    'max_tokens': 1200,
    'temperature': 0.7  # 응답의 창의성 조절
}

try:
    response = get_completion(**data)
    print(response.choices[0].message.content.strip())
except RateLimitError as e:
    print("다시 시도해주세요")
    time.sleep(60)  # 1분 대기 후 재시도
except OpenAIError as e:
    print(f"An error occurred: {e}")

print()
print('선택한 조건에 맞는 공장 부지를 추천합니다.')
# 조건에 맞는 행 필터링
filtered_factories = Factories[
    (Factories['매물위치(1)'] == korean_name) & 
    (Factories['매물위치(3)'] == predicted_class)
]

# 필요한 열만 선택하여 필터링된 데이터프레임 생성
columns_to_display = ['매물위치(1)', '매물위치(2)', '매물위치(3)', '종류/장점', '제목', '상태', '가격', '면적', '방식', '참고', '조회수', '매물위치(url)']
filtered_factories_to_display = filtered_factories[columns_to_display]

# 필터링된 데이터가 있는지 확인
if filtered_factories_to_display.empty:
    print("현재 매물이 존재하지 않습니다. 선택한 지자체 관련된 모든 공장 데이터를 제공합니다.")
    filtered_factories = Factories[
        (Factories['매물위치(1)'] == korean_name)
    ]
    filtered_factories_to_display = filtered_factories[columns_to_display]

# DataFrame을 콘솔에 출력
print(filtered_factories_to_display)
print('추천을 완료하였습니다. csv 파일로 저장 완료하였습니다.')

# 결과를 CSV 파일로 저장
filtered_factories_to_display.to_csv(f'{korean_name}_{predicted_class}_filtered_factories.csv', index=False, encoding='cp949')


