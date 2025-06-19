import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
import os

# --- 앱 설정 ---
st.set_page_config(page_title="⚖️ 법률 전문 번역기", page_icon="⚖️")

# --- 모델 로딩 (수정된 부분) ---
# Hugging Face Hub에 있는 본인의 모델 저장소 주소를 정확하게 입력합니다.
KO_EN_MODEL_PATH = "lcm52/legal-marian-koen"
EN_KO_MODEL_PATH = "lcm52/legal-marian-enko"

@st.cache_resource
def load_models():
    """
    파인튜닝된 한-영, 영-한 번역 모델과 토크나이저를 로드합니다.
    """
    # --- 한-영 모델 로드 ---
    try:
        ko_en_tokenizer = MarianTokenizer.from_pretrained(KO_EN_MODEL_PATH)
        ko_en_model = MarianMTModel.from_pretrained(KO_EN_MODEL_PATH)
    except OSError:
        st.error(f"오류: 한국어->영어 모델 로드에 실패했습니다. Hugging Face Hub에서 '{KO_EN_MODEL_PATH}' 저장소를 찾을 수 없습니다. 저장소 이름과 공개(Public) 상태를 확인하세요.")
        return None

    # --- 영-한 모델 로드 ---
    try:
        en_ko_tokenizer = MarianTokenizer.from_pretrained(EN_KO_MODEL_PATH)
        en_ko_model = MarianMTModel.from_pretrained(EN_KO_MODEL_PATH)
    except OSError:
        st.error(f"오류: 영어->한국어 모델 로드에 실패했습니다. Hugging Face Hub에서 '{EN_KO_MODEL_PATH}' 저장소를 찾을 수 없습니다. 저장소 이름과 공개(Public) 상태를 확인하세요.")
        return None
        
    models = {
        "ko_en": (ko_en_tokenizer, ko_en_model),
        "en_ko": (en_ko_tokenizer, en_ko_model)
    }
    return models

def translate_text(text, tokenizer, model):
    """
    입력된 텍스트를 주어진 모델과 토크나이저를 사용하여 번역합니다.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = model.generate(**inputs, max_length=512)
    result = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return result

# --- UI (사용자 인터페이스) ---
st.title("⚖️ 법률 전문 번역기")
st.caption("대한민국 민법 데이터로 파인튜닝된 MarianMT 모델 기반")

direction = st.radio(
    "번역할 방향을 선택하세요:",
    ("한국어 → 영어", "영어 → 한국어"),
    horizontal=True,
)

input_text = st.text_area("번역할 법률 문장을 입력하세요...", height=150)

if st.button("번역하기", type="primary"):
    if not input_text.strip():
        st.warning("번역할 문장을 입력해주세요.")
    else:
        loaded_models = load_models()
        if loaded_models:
            if direction == "한국어 → 영어":
                tokenizer, model = loaded_models["ko_en"]
            else:
                tokenizer, model = loaded_models["en_ko"]
            
            with st.spinner("전문 용어를 분석하며 번역 중입니다... 잠시만 기다려주세요."):
                translated_result = translate_text(input_text, tokenizer, model)

            st.divider()
            st.subheader("번역 결과", anchor=False)
            st.markdown(f"> {translated_result}")

st.divider()
st.info("이 앱은 교육 및 데모 목적으로 제작되었으며, 실제 법적 효력을 갖는 번역을 제공하지 않습니다.")
