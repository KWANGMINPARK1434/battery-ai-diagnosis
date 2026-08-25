import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Battery Process AI Diagnosis System", layout="wide")

st.title("🔋 배터리 공정 불량 원인 진단 & AI 트러블슈팅 플랫폼")
st.caption("3,500+ Raw Data 기반 타 조/불량 셀 패턴 분석 및 공정 피드백 파이프라인")

# 1. Excel 데이터 로드 (조별 데이터 파싱)
@st.cache_data
def load_all_groups():
    excel_files = glob.glob("*.xlsx")
    cell_file = None
    for f in excel_files:
        if "충방전" in f or "cell" in f.lower() or "12주차" in f:
            cell_file = f
            break
    if not cell_file and excel_files:
        cell_file = excel_files[0]
        
    if cell_file:
        df_raw = pd.read_excel(cell_file, sheet_name='조별 Cell data')
        groups_data = {}
        col_pairs = [(0,1), (3,4), (6,7), (12,13), (18,19), (21,22)]
        for idx, (c_cap, c_vol) in enumerate(col_pairs, start=1):
            sub_df = df_raw.iloc[5:, [c_cap, c_vol]].dropna().reset_index(drop=True)
            sub_df.columns = ['capacity', 'voltage']
            groups_data[f"{idx}조"] = sub_df.astype(float)
        return groups_data, cell_file
    else:
        np.random.seed(42)
        groups_data = {}
        for g in range(1, 7):
            cap = np.linspace(0, 160, 3500)
            noise = np.random.normal(0, 0.005 if g != 5 else 0.02, 3500)
            ir_factor = 0.002 if g != 5 else 0.008
            vol = 4.2 - (cap/160)**0.5 * 1.2 - (cap * ir_factor) + noise
            groups_data[f"{g}조"] = pd.DataFrame({'capacity': cap, 'voltage': vol})
        return groups_data, "Simulation_MultiGroup_Engine"

groups_dict, file_src = load_all_groups()

st.sidebar.header("🎯 공정 데이터 분석 타겟")
st.sidebar.text_input("데이터 소스", file_src, disabled=True)

selected_group = st.sidebar.selectbox("분석 대상 선택 (실패/불량 조 검증)", list(groups_dict.keys()), index=4)
df_target = groups_dict[selected_group].copy()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ AI 파라미터")
contamination = st.sidebar.slider("이상치 탐지 민감도 (Isolation Forest)", 0.005, 0.05, 0.02, step=0.005)

# 전처리 & 특징 추출
df_target['voltage_smooth'] = df_target['voltage'].rolling(window=5, center=True).mean().fillna(df_target['voltage'])
df_target['dQ'] = df_target['capacity'].diff().replace(0, np.nan)
df_target['dV'] = df_target['voltage_smooth'].diff()
df_target['dV_dQ'] = (df_target['dV'] / df_target['dQ']).abs()
df_target['dQ_dV'] = (df_target['dQ'] / df_target['dV']).abs()

df_clean = df_target[(df_target['dV_dQ'] < 15) & (df_target['dV_dQ'] > 0)].dropna()

# Isolation Forest ML
features = df_clean[['capacity', 'voltage_smooth', 'dV_dQ']]
model = IsolationForest(contamination=contamination, random_state=42)
df_clean['anomaly'] = model.fit_predict(features)

anom_cnt = (df_clean['anomaly'] == -1).sum()
max_cap = df_clean['capacity'].max()
target_cap = 150.0
cap_retention = (max_cap / target_cap) * 100
avg_ir = df_clean['dV_dQ'].quantile(0.90)

lli_val = max(0.0, round((100 - cap_retention) * 0.6, 1))
lam_val = max(0.0, round((100 - cap_retention) * 0.3, 1))
ri_val = min(100.0, round(avg_ir * 15.0, 1))

if cap_retention < 90.0 and avg_ir > 3.0:
    primary_failure = "복합 결함 (LLI + RI)"
elif cap_retention < 92.0:
    primary_failure = "LLI (리튬 이온 손실)"
elif avg_ir > 3.5:
    primary_failure = "RI (내부 저항 증가 및 IR Drop)"
elif anom_cnt > 80:
    primary_failure = "LAM (슬러리 코팅 불균일/활물질 탈락)"
else:
    primary_failure = "정상 (Pass)"

st.header(f"🔍 {selected_group} Cell Data AI 진단 결과")

col1, col2, col3, col4 = st.columns(4)
col1.metric("최대 용량 발현율", f"{cap_retention:.1f} %")
col2.metric("이상치 탐지 수", f"{anom_cnt} 개")
col3.metric("추정 내부저항 지표", f"{avg_ir:.2f}")
col4.metric("주요 불량 판단", primary_failure)

st.markdown("---")

tab1, tab2 = st.tabs(["📊 V-Q 및 dQ/dV 분석 스펙트럼", "🛠️ AI 불량 원인 분석 및 공정 피드백 가이드"])

with tab1:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{selected_group} V-Q Pattern & Anomalies", f"{selected_group} dQ/dV Differential Peak"))
    
    norm_df = df_clean[df_clean['anomaly'] == 1]
    anom_df = df_clean[df_clean['anomaly'] == -1]
    
    fig.add_trace(go.Scatter(x=norm_df['capacity'], y=norm_df['voltage_smooth'], mode='lines', name='정상 데이터', line=dict(color='navy')), row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_df['capacity'], y=anom_df['voltage_smooth'], mode='markers', name='이상 지점', marker=dict(color='red', size=6, symbol='x')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean['voltage_smooth'], y=df_clean['dQ_dV'], mode='lines', name='dQ/dV Peak', line=dict(color='teal')), row=1, col=2)
    
    fig.update_layout(height=420, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        st.subheader("📊 불량 원인 비중 정량화")
        st.progress(min(int(lli_val), 100), text=f"LLI (리튬 손실): {lli_val}%")
        st.progress(min(int(lam_val), 100), text=f"LAM (활물질 손실): {lam_val}%")
        st.progress(min(int(ri_val), 100), text=f"RI (저항 증가): {ri_val}%")
        
    with col_a2:
        st.subheader("📋 Root Cause 원인 분석 & 트러블슈팅 가이드")
        
        if "LLI" in primary_failure:
            st.error("🚨 **[진단: 리튬 이온 손실(LLI) 발생]** Formation 초기 SEI 피막 과다 형성 문제")
            st.markdown("""
            * **원인 추정:** 초기 충전 C-rate가 높거나 전해액 주입 후 Aging 시간 부족으로 비가역 리튬 소모 증대.
            * **공정 개선 가이드:**
              1. **Formation 레시피 수정:** Initial Charge 속도를 0.1C -> 0.05C로 하향 조정.
              2. **Aging 공정 점검:** High-Temp Aging 온도를 45℃ 기준 ±1℃ 범위로 정밀 제어.
            """)
        elif "RI" in primary_failure:
            st.warning("⚠️ **[진단: 내부 저항(RI) 상승]** 셀 접촉 저항 및 탭 용접 불량")
            st.markdown("""
            * **원인 추정:** Ultra-sonic Tab Welding 출력 부족 또는 전극 코팅면 탈락으로 인한 저항 증가.
            * **공정 개선 가이드:**
              1. **설비 점검:** 초음파 용접기 Horn 및 Anvil 마모도 측정 후 가압력 5% 증량.
              2. **전해액 점검:** 전해액 함침(Wetting) 시간을 15분 연장하여 내부 저항 감소 도모.
            """)
        elif "LAM" in primary_failure:
            st.warning("⚠️ **[진단: 활물질 손실(LAM)]** 슬러리 분산 및 코팅 두께 불균일")
            st.markdown("""
            * **원인 추정:** 믹싱 공정 바인더 분산 미흡 또는 Roll-to-Roll 코터 닥터블레이드 Gap 편차.
            * **공정 개선 가이드:**
              1. **슬러리 믹싱:** PD Mixer 교반 속도 증대 및 점도(Viscosity) 상한선 스펙 재설정.
              2. **코팅 공정:** 롤투롤 코팅 두께 프로파일 실시간 센서 보정.
            """)
        else:
            st.success("✅ **[진단: 공정 정상 (Pass)]** 해당 조의 셀 성능 및 반응 패턴이 기준 스펙을 충족합니다.")
            st.markdown("* **공정 유지 가이드:** 현행 슬러리 코팅 Gap, 주입량, Formation 조건 단일성 유지.")
