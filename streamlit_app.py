import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from snowflake.snowpark.context import get_active_session

session = get_active_session()

st.set_page_config(page_title="부동산 FGI 분석 대시보드", layout="wide")

@st.cache_resource
def load_registry_models():
    try:
        from snowflake.ml.registry import Registry
        reg = Registry(session=session, database_name='ML_REGISTRY', schema_name='PUBLIC')
        analysis_model = reg.get_model('FGI_ANALYSIS_RF').version('v1')
        predict_model = reg.get_model('FGI_PREDICT_XGB').version('v1')
        return analysis_model, predict_model, True
    except Exception:
        return None, None, False

analysis_mv, predict_mv, registry_available = load_registry_models()

FEATURE_LABELS = {
    "CHANGE_MEME_PRICE_RATE": "매매가 변동률",
    "REVERSE_JEONSE_PER_MEME": "역전세율",
    "RATE_GAP": "금리갭",
    "PRICE_MOMENTUM_3M": "3개월 모멘텀",
    "PRICE_VOLATILITY_6M": "6개월 변동성",
    "MOVEMENT_RATE": "이동률",
    "CONTRACT_COUNT": "거래건수",
    "MEAN_MEME_PRICE": "평균 매매가",
    "MEAN_JEONSE_PRICE": "평균 전세가",
}

FEATURES = ["CHANGE_MEME_PRICE_RATE", "REVERSE_JEONSE_PER_MEME", "RATE_GAP",
            "PRICE_MOMENTUM_3M", "PRICE_VOLATILITY_6M", "MOVEMENT_RATE"]


def normalize_fgi(df):
    result = df.copy()
    result["FGI_RAW"] = result["FEAR_GREED_INDEX"]
    state_stats = result.groupby("STATE")["FEAR_GREED_INDEX"].agg(["mean", "std"]).reset_index()
    state_stats.columns = ["STATE", "FGI_STATE_MEAN", "FGI_STATE_STD"]
    result = result.merge(state_stats, on="STATE", how="left")
    result["FGI_STATE_STD"] = result["FGI_STATE_STD"].replace(0, 1)
    result["FGI_ZSCORE"] = (result["FEAR_GREED_INDEX"] - result["FGI_STATE_MEAN"]) / result["FGI_STATE_STD"]
    result["FGI_NORM"] = 50 + 50 * result["FGI_ZSCORE"].clip(-2, 2) / 2
    return result


def fgi_label(val):
    if val <= 20:
        return "극도의 공포", "#d32f2f", "😱"
    elif val <= 40:
        return "공포", "#f57c00", "😟"
    elif val <= 60:
        return "중립", "#fbc02d", "😐"
    elif val <= 80:
        return "탐욕", "#388e3c", "😊"
    else:
        return "극도의 탐욕", "#1b5e20", "🤑"


def fgi_card(label, value, fgi_val):
    lbl, clr, emoji = fgi_label(fgi_val)
    return f"""
    <div style="background:{clr}15;border-left:5px solid {clr};border-radius:8px;padding:16px 20px;text-align:center;">
        <p style="margin:0;font-size:0.85em;color:#666;">{label}</p>
        <p style="margin:4px 0;font-size:2em;font-weight:bold;color:{clr};">{value}</p>
        <p style="margin:0;font-size:1.1em;color:{clr};font-weight:600;">{emoji} {lbl}</p>
    </div>
    """


def signal_label(row):
    fgi_n = row.get("FGI_NORM", 50)
    trend = row.get("CONTRACT_TREND_NORM", None)
    if trend is not None and trend > 5 and fgi_n > 65:
        return "Opportunity"
    elif trend is not None and trend < 1 and fgi_n < 35:
        return "Warning"
    return "Stable"


@st.cache_data(ttl=600)
def load_fgi():
    return session.sql(
        "SELECT * FROM PRICE_BASE.PUBLIC.CALC_FGI_INTERNET_NN_MEAN ORDER BY YEAR_MONTH"
    ).to_pandas()


@st.cache_data(ttl=600)
def load_contract():
    return session.sql(
        "SELECT * FROM CONTRACT_TRENDS.PUBLIC.CONTRACT_TREND_NORMALIZED ORDER BY YEAR_MONTH"
    ).to_pandas()


@st.cache_data(ttl=600)
def load_all_fgi():
    tables = {
        "기본": "CALC_FGI",
        "인터넷반영": "CALC_FGI_INTERNET",
        "NN보정": "CALC_FGI_INTERNET_NN",
        "NN+평균": "CALC_FGI_INTERNET_NN_MEAN",
    }
    frames = {}
    for lbl, tbl in tables.items():
        frames[lbl] = session.sql(
            f"SELECT YEAR_MONTH, STATE, CITY, FEAR_GREED_INDEX FROM PRICE_BASE.PUBLIC.{tbl}"
        ).to_pandas()
    return frames


df_fgi_raw = load_fgi()
df_contract = load_contract()

df_fgi_raw["YEAR_MONTH"] = pd.to_datetime(df_fgi_raw["YEAR_MONTH"])
df_contract["YEAR_MONTH"] = pd.to_datetime(df_contract["YEAR_MONTH"])

df_fgi = normalize_fgi(df_fgi_raw)

latest_fgi_date = df_fgi["YEAR_MONTH"].max()
prev_fgi_date = df_fgi[df_fgi["YEAR_MONTH"] < latest_fgi_date]["YEAR_MONTH"].max()

latest_fgi = df_fgi[df_fgi["YEAR_MONTH"] == latest_fgi_date].copy()
prev_fgi = df_fgi[df_fgi["YEAR_MONTH"] == prev_fgi_date].copy()

today = pd.Timestamp("today").normalize()
ct_max_date = df_contract["YEAR_MONTH"].max()
if ct_max_date.month == today.month and ct_max_date.year == today.year:
    ct_dates_sorted = sorted(df_contract["YEAR_MONTH"].unique(), reverse=True)
    latest_ct_date = ct_dates_sorted[1] if len(ct_dates_sorted) > 1 else ct_max_date
else:
    latest_ct_date = ct_max_date
prev_ct_date = df_contract[df_contract["YEAR_MONTH"] < latest_ct_date]["YEAR_MONTH"].max()
latest_ct = df_contract[df_contract["YEAR_MONTH"] == latest_ct_date].copy()
prev_ct = df_contract[df_contract["YEAR_MONTH"] == prev_ct_date].copy()

merged_latest = latest_fgi.merge(
    latest_ct, left_on=["STATE", "CITY"], right_on=["INSTALL_STATE", "INSTALL_CITY"], how="inner",
    suffixes=("", "_CT")
)
merged_prev = prev_fgi.merge(
    prev_ct, left_on=["STATE", "CITY"], right_on=["INSTALL_STATE", "INSTALL_CITY"], how="inner",
    suffixes=("", "_CT")
)

states = sorted(df_fgi["STATE"].unique())

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Snowflake_Logo.svg/200px-Snowflake_Logo.svg.png", width=160)
    st.title("필터 설정")
    sel_state = st.selectbox("시/도", states, index=states.index("서울") if "서울" in states else 0)
    cities = sorted(df_fgi[df_fgi["STATE"] == sel_state]["CITY"].unique())
    sel_city = st.selectbox("시/군/구", cities)
    min_date = df_fgi["YEAR_MONTH"].min().date()
    max_date = df_fgi["YEAR_MONTH"].max().date()
    date_range = st.date_input("기간", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.markdown("---")
    st.caption(f"FGI 데이터: {min_date} ~ {max_date}")
    st.caption(f"계약 데이터: {df_contract['YEAR_MONTH'].min().date()} ~ {latest_ct_date.date()}")

if len(date_range) == 2:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_dt, end_dt = pd.Timestamp(min_date), pd.Timestamp(max_date)

st.markdown(
    "<h1 style='text-align:center;'>부동산 시장 심리 분석 대시보드</h1>"
    "<p style='text-align:center;color:gray;'>FGI(Fear-Greed Index) 기반 의사결정 지원 시스템</p>",
    unsafe_allow_html=True,
)

tab_overview, tab_region, tab_signal, tab_ask = st.tabs([
    "📊 Market Overview", "🔍 Regional Deep Dive", "🚦 Signal & Prediction", "💬 Ask Data"
])

# ──────────────────────────────────────────────
# TAB 1: MARKET OVERVIEW
# ──────────────────────────────────────────────
with tab_overview:
    st.subheader("전체 시장 요약")

    avg_fgi_norm = latest_fgi["FGI_NORM"].mean()
    avg_fgi_raw = latest_fgi["FGI_RAW"].mean()

    ct_compare = latest_ct.merge(
        prev_ct[["INSTALL_STATE", "INSTALL_CITY", "CONTRACT_COUNT"]],
        on=["INSTALL_STATE", "INSTALL_CITY"], how="inner", suffixes=("", "_PREV")
    )
    increase_count = int((ct_compare["CONTRACT_COUNT"] > ct_compare["CONTRACT_COUNT_PREV"]).sum())
    decrease_count = int((ct_compare["CONTRACT_COUNT"] < ct_compare["CONTRACT_COUNT_PREV"]).sum())

    k1, k2, k3 = st.columns(3)
    k1.markdown(fgi_card("전국 평균 FGI", f"{avg_fgi_norm:.1f}", avg_fgi_norm), unsafe_allow_html=True)
    k2.markdown(f"""
    <div style="background:#e8f5e915;border-left:5px solid #4caf50;border-radius:8px;padding:16px 20px;text-align:center;">
        <p style="margin:0;font-size:0.85em;color:#666;">계약 증가 지역</p>
        <p style="margin:4px 0;font-size:2em;font-weight:bold;color:#4caf50;">{increase_count}개</p>
        <p style="margin:0;font-size:1.1em;color:#4caf50;">▲ 전월 대비</p>
    </div>
    """, unsafe_allow_html=True)
    k3.markdown(f"""
    <div style="background:#ffebee15;border-left:5px solid #f44336;border-radius:8px;padding:16px 20px;text-align:center;">
        <p style="margin:0;font-size:0.85em;color:#666;">계약 감소 지역</p>
        <p style="margin:4px 0;font-size:2em;font-weight:bold;color:#f44336;">{decrease_count}개</p>
        <p style="margin:0;font-size:1.1em;color:#f44336;">▼ 전월 대비</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_top, col_bot = st.columns(2)

    with col_top:
        st.markdown("##### 계약 증가 가능성 Top 5")
        if not merged_latest.empty:
            top5_up = merged_latest.nlargest(5, "CONTRACT_TREND_NORM")[
                ["STATE", "CITY", "FGI_NORM", "FGI_RAW", "CONTRACT_TREND_NORM", "CONTRACT_COUNT"]
            ].reset_index(drop=True)
            top5_up.columns = ["시/도", "시/군/구", "FGI(정규화)", "FGI(원본)", "트렌드점수", "거래건수"]
            top5_up["FGI(정규화)"] = top5_up["FGI(정규화)"].round(1)
            top5_up["FGI(원본)"] = top5_up["FGI(원본)"].round(0)
            top5_up["트렌드점수"] = top5_up["트렌드점수"].round(2)
            st.dataframe(top5_up, hide_index=True, width=600)

    with col_bot:
        st.markdown("##### 계약 감소 위험 Top 5")
        if not merged_latest.empty:
            top5_down = merged_latest.nsmallest(5, "CONTRACT_TREND_NORM")[
                ["STATE", "CITY", "FGI_NORM", "FGI_RAW", "CONTRACT_TREND_NORM", "CONTRACT_COUNT"]
            ].reset_index(drop=True)
            top5_down.columns = ["시/도", "시/군/구", "FGI(정규화)", "FGI(원본)", "트렌드점수", "거래건수"]
            top5_down["FGI(정규화)"] = top5_down["FGI(정규화)"].round(1)
            top5_down["FGI(원본)"] = top5_down["FGI(원본)"].round(0)
            top5_down["트렌드점수"] = top5_down["트렌드점수"].round(2)
            st.dataframe(top5_down, hide_index=True, width=600)

    st.markdown("---")
    st.markdown("##### 시/도별 FGI 히트맵")

    state_agg = df_fgi.groupby(["YEAR_MONTH", "STATE"])["FGI_NORM"].mean().reset_index()
    state_agg = state_agg[(state_agg["YEAR_MONTH"] >= start_dt) & (state_agg["YEAR_MONTH"] <= end_dt)]
    state_agg["YM"] = state_agg["YEAR_MONTH"].dt.strftime("%Y-%m")

    heatmap = alt.Chart(state_agg).mark_rect().encode(
        x=alt.X("YM:O", title="기간", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("STATE:N", title="시/도"),
        color=alt.Color("FGI_NORM:Q", title="FGI (정규화)",
                         scale=alt.Scale(scheme="redyellowgreen", domain=[0, 100])),
        tooltip=[
            alt.Tooltip("STATE:N", title="시/도"),
            alt.Tooltip("YM:O", title="기간"),
            alt.Tooltip("FGI_NORM:Q", title="FGI (정규화)", format=".1f"),
        ],
    ).properties(height=450)
    st.altair_chart(heatmap, width="stretch")

# ──────────────────────────────────────────────
# TAB 2: REGIONAL DEEP DIVE
# ──────────────────────────────────────────────
with tab_region:
    st.subheader(f"📍 {sel_state} {sel_city} 상세 분석")

    city_fgi = df_fgi[
        (df_fgi["STATE"] == sel_state) & (df_fgi["CITY"] == sel_city) &
        (df_fgi["YEAR_MONTH"] >= start_dt) & (df_fgi["YEAR_MONTH"] <= end_dt)
    ].copy()

    city_ct = df_contract[
        (df_contract["INSTALL_STATE"] == sel_state) & (df_contract["INSTALL_CITY"] == sel_city) &
        (df_contract["YEAR_MONTH"] >= start_dt) & (df_contract["YEAR_MONTH"] <= end_dt)
    ].copy()

    if not city_fgi.empty:
        latest_row = city_fgi.iloc[-1]
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(fgi_card("FGI (정규화)", f"{latest_row['FGI_NORM']:.1f}", latest_row["FGI_NORM"]), unsafe_allow_html=True)
        r2.metric("매매가", f"{latest_row['MEAN_MEME_PRICE']:,.0f}만원")
        r3.metric("전세가", f"{latest_row['MEAN_JEONSE_PRICE']:,.0f}만원")
        r4.metric("거래건수", f"{int(latest_row['CONTRACT_COUNT']):,}")

    st.markdown("---")
    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("##### 계약 건수 추이")
        if not city_ct.empty:
            ct_chart = alt.Chart(city_ct).mark_line(color="#1976d2", strokeWidth=2).encode(
                x=alt.X("YEAR_MONTH:T", title="기간"),
                y=alt.Y("CONTRACT_COUNT:Q", title="거래건수"),
                tooltip=[alt.Tooltip("YEAR_MONTH:T", format="%Y-%m"), alt.Tooltip("CONTRACT_COUNT:Q", format=",")],
            ).properties(height=300)
            st.altair_chart(ct_chart, width="stretch")
        else:
            st.info("해당 지역의 계약 데이터가 없습니다.")

    with rc2:
        st.markdown("##### FGI 추이")
        if not city_fgi.empty:
            fgi_line = alt.Chart(city_fgi).mark_area(opacity=0.3, line={"color": "#e65100"}).encode(
                x=alt.X("YEAR_MONTH:T", title="기간"),
                y=alt.Y("FGI_NORM:Q", title="FGI (정규화, 0~100)", scale=alt.Scale(domain=[0, 100])),
                color=alt.value("#e65100"),
                tooltip=[alt.Tooltip("YEAR_MONTH:T", format="%Y-%m"), alt.Tooltip("FGI_NORM:Q", title="FGI(정규화)", format=".1f"), alt.Tooltip("FGI_RAW:Q", title="FGI(원본)", format=".0f")],
            ).properties(height=300)
            rule_50 = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart(fgi_line + rule_50, width="stretch")

    st.markdown("---")
    st.markdown("##### FGI vs 계약 트렌드 비교")

    if not city_fgi.empty and not city_ct.empty:
        merged_city = city_fgi[["YEAR_MONTH", "FGI_NORM"]].merge(
            city_ct[["YEAR_MONTH", "CONTRACT_TREND_NORM"]], on="YEAR_MONTH", how="inner"
        )
        if not merged_city.empty:
            fgi_norm_vals = merged_city["FGI_NORM"] / 100

            ct_norm_vals = (merged_city["CONTRACT_TREND_NORM"] - merged_city["CONTRACT_TREND_NORM"].min())
            ct_range = merged_city["CONTRACT_TREND_NORM"].max() - merged_city["CONTRACT_TREND_NORM"].min()
            if ct_range > 0:
                ct_norm_vals = ct_norm_vals / ct_range
            else:
                ct_norm_vals = ct_norm_vals * 0

            compare_df = pd.DataFrame({
                "YEAR_MONTH": list(merged_city["YEAR_MONTH"]) * 2,
                "value": list(fgi_norm_vals) + list(ct_norm_vals),
                "지표": ["FGI (정규화)"] * len(merged_city) + ["계약트렌드 (정규화)"] * len(merged_city),
            })
            dual_chart = alt.Chart(compare_df).mark_line(strokeWidth=2).encode(
                x=alt.X("YEAR_MONTH:T", title="기간"),
                y=alt.Y("value:Q", title="정규화 값 (0~1)"),
                color=alt.Color("지표:N", scale=alt.Scale(domain=["FGI (정규화)", "계약트렌드 (정규화)"],
                                                          range=["#e65100", "#1565c0"])),
                tooltip=[alt.Tooltip("YEAR_MONTH:T", format="%Y-%m"), alt.Tooltip("value:Q", format=".3f"), "지표:N"],
            ).properties(height=350)
            st.altair_chart(dual_chart, width="stretch")

    st.markdown("---")
    st.markdown("##### FGI 모델 비교 (실제값 vs 변형)")
    fgi_frames = load_all_fgi()
    compare_rows = []
    for lbl, frame in fgi_frames.items():
        frame["YEAR_MONTH"] = pd.to_datetime(frame["YEAR_MONTH"])
        subset = frame[
            (frame["STATE"] == sel_state) & (frame["CITY"] == sel_city) &
            (frame["YEAR_MONTH"] >= start_dt) & (frame["YEAR_MONTH"] <= end_dt)
        ].copy()
        if not subset.empty:
            subset["모델"] = lbl
            compare_rows.append(subset[["YEAR_MONTH", "FEAR_GREED_INDEX", "모델"]])

    if compare_rows:
        comp_df = pd.concat(compare_rows, ignore_index=True)
        comp_chart = alt.Chart(comp_df).mark_line(strokeWidth=2).encode(
            x=alt.X("YEAR_MONTH:T", title="기간"),
            y=alt.Y("FEAR_GREED_INDEX:Q", title="FGI"),
            color=alt.Color("모델:N"),
            strokeDash=alt.StrokeDash("모델:N"),
            tooltip=[alt.Tooltip("YEAR_MONTH:T", format="%Y-%m"), "모델:N",
                     alt.Tooltip("FEAR_GREED_INDEX:Q", format=".1f")],
        ).properties(height=350)
        st.altair_chart(comp_chart, width="stretch")

    st.markdown("---")
    st.markdown("##### 변수 영향도 분석")

    if registry_available and analysis_mv is not None:
        try:
            features_analysis = ["CONTRACT_COUNT", "MOVEMENT_RATE", "MEAN_MEME_PRICE", "MEAN_JEONSE_PRICE",
                                 "CHANGE_MEME_PRICE_RATE", "REVERSE_JEONSE_PER_MEME", "RATE_GAP",
                                 "PRICE_MOMENTUM_3M", "PRICE_VOLATILITY_6M"]
            if not city_fgi.empty:
                sample_input = city_fgi[features_analysis].tail(10).reset_index(drop=True)
                sp_df = session.create_dataframe(sample_input)
                preds = analysis_mv.run(sp_df, function_name='predict').to_pandas()

                metrics = analysis_mv.show_metrics()
                st.success(f"Registry 모델 (FGI_ANALYSIS_RF v1) 연동 | R2: {metrics.get('R2', 'N/A')}, RMSE: {metrics.get('RMSE', 'N/A')}")

                st.markdown("**실제값 vs 모델 예측값**")
                actual_vs_pred = pd.DataFrame({
                    "YEAR_MONTH": city_fgi["YEAR_MONTH"].tail(10).values,
                    "실제값": city_fgi["CONTRACT_COUNT"].tail(10).values,
                    "예측값": preds.iloc[:, -1].values,
                })
                avp_melt = actual_vs_pred.melt(id_vars="YEAR_MONTH", var_name="유형", value_name="거래건수")
                avp_chart = alt.Chart(avp_melt).mark_line(strokeWidth=2).encode(
                    x=alt.X("YEAR_MONTH:T", title="기간"),
                    y=alt.Y("거래건수:Q"),
                    color=alt.Color("유형:N", scale=alt.Scale(domain=["실제값", "예측값"], range=["#1976d2", "#e53935"])),
                    strokeDash=alt.StrokeDash("유형:N"),
                    tooltip=[alt.Tooltip("YEAR_MONTH:T", format="%Y-%m"), "유형:N", alt.Tooltip("거래건수:Q", format=",.0f")],
                ).properties(height=300)
                st.altair_chart(avp_chart, width="stretch")
        except Exception as e:
            st.warning(f"Registry 모델 호출 실패: {e}")
            st.caption("상관계수 기반 영향도로 대체합니다.")
            registry_available_fallback = True
    else:
        registry_available_fallback = True

    if not registry_available or 'registry_available_fallback' in dir():
        if not city_fgi.empty and len(city_fgi) > 3:
            avail_features = [f for f in FEATURES if f in city_fgi.columns]
            importance = {}
            for feat in avail_features:
                series = city_fgi[feat].astype(float)
                fgi_series = city_fgi["FEAR_GREED_INDEX"].astype(float)
                if series.std() > 0 and fgi_series.std() > 0:
                    importance[FEATURE_LABELS.get(feat, feat)] = abs(series.corr(fgi_series))
                else:
                    importance[FEATURE_LABELS.get(feat, feat)] = 0.0

            imp_df = pd.DataFrame({"변수": list(importance.keys()), "영향도": list(importance.values())})
            imp_df = imp_df.sort_values("영향도", ascending=False).head(5)

            imp_chart = alt.Chart(imp_df).mark_bar(color="#5e35b1").encode(
                x=alt.X("영향도:Q", title="상관계수 (절대값)"),
                y=alt.Y("변수:N", sort="-x", title=""),
                tooltip=[alt.Tooltip("변수:N"), alt.Tooltip("영향도:Q", format=".3f")],
            ).properties(height=250)
            st.altair_chart(imp_chart, width="stretch")
            st.caption("FGI와 각 변수 간의 피어슨 상관계수 절대값 기반 영향도")
        else:
            st.info("데이터가 충분하지 않아 변수 영향도를 계산할 수 없습니다.")

# ──────────────────────────────────────────────
# TAB 3: SIGNAL & PREDICTION
# ──────────────────────────────────────────────
with tab_signal:
    st.subheader("시장 신호 탐지")

    if registry_available and predict_mv is not None:
        st.markdown("##### 🤖 ML 모델 실시간 예측 (XGBoost)")
        try:
            pred_metrics = predict_mv.show_metrics()
            st.info(f"FGI_PREDICT_XGB v1 | R2: {pred_metrics.get('R2', 'N/A')}, RMSE: {pred_metrics.get('RMSE', 'N/A')}")
        except Exception:
            pass
        st.caption("Registry에 등록된 XGBoost 예측모델이 연동되어 있습니다. 아래 신호 분석과 함께 활용됩니다.")
        st.markdown("---")

    if not merged_latest.empty:
        scored = merged_latest.copy()

        scored["FGI_SCORE"] = scored["FGI_NORM"] / 2

        ct_min = scored["CONTRACT_TREND_NORM"].min()
        ct_max = scored["CONTRACT_TREND_NORM"].max()
        ct_range_s = ct_max - ct_min if ct_max != ct_min else 1
        scored["CT_SCORE"] = (scored["CONTRACT_TREND_NORM"] - ct_min) / ct_range_s * 50

        scored["SIGNAL_SCORE"] = scored["FGI_SCORE"] + scored["CT_SCORE"]
        scored["상태"] = scored.apply(signal_label, axis=1)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("##### 🟢 계약 증가 가능성 Top 10 (Opportunity)")
            top10_up = scored.nlargest(10, "SIGNAL_SCORE")[
                ["STATE", "CITY", "FGI_NORM", "CONTRACT_TREND_NORM", "SIGNAL_SCORE", "상태"]
            ].reset_index(drop=True)
            top10_up.columns = ["시/도", "시/군/구", "FGI", "트렌드", "점수", "상태"]
            top10_up["FGI"] = top10_up["FGI"].round(1)
            top10_up["트렌드"] = top10_up["트렌드"].round(2)
            top10_up["점수"] = top10_up["점수"].round(1)
            st.dataframe(top10_up, hide_index=True, width=550)

        with s2:
            st.markdown("##### 🔴 계약 감소 위험 Top 10 (Warning)")
            top10_down = scored.nsmallest(10, "SIGNAL_SCORE")[
                ["STATE", "CITY", "FGI_NORM", "CONTRACT_TREND_NORM", "SIGNAL_SCORE", "상태"]
            ].reset_index(drop=True)
            top10_down.columns = ["시/도", "시/군/구", "FGI", "트렌드", "점수", "상태"]
            top10_down["FGI"] = top10_down["FGI"].round(1)
            top10_down["트렌드"] = top10_down["트렌드"].round(2)
            top10_down["점수"] = top10_down["점수"].round(1)
            st.dataframe(top10_down, hide_index=True, width=550)

        st.markdown("---")
        st.markdown("##### 전체 지역 신호 분포")

        signal_counts = scored["상태"].value_counts().reset_index()
        signal_counts.columns = ["상태", "지역수"]
        color_map = {"Opportunity": "#4caf50", "Warning": "#f44336", "Stable": "#ff9800"}

        pie_chart = alt.Chart(signal_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("지역수:Q"),
            color=alt.Color("상태:N", scale=alt.Scale(
                domain=list(color_map.keys()), range=list(color_map.values())
            )),
            tooltip=["상태:N", "지역수:Q"],
        ).properties(height=300, width=300)
        st.altair_chart(pie_chart, width="stretch")

        st.markdown("---")
        st.markdown("##### 지역별 신호 점수 분포")
        scatter = alt.Chart(scored).mark_circle(size=60).encode(
            x=alt.X("FGI_NORM:Q", title="FGI (정규화, 0~100)"),
            y=alt.Y("CONTRACT_TREND_NORM:Q", title="계약 트렌드 (정규화)"),
            color=alt.Color("상태:N", scale=alt.Scale(
                domain=list(color_map.keys()), range=list(color_map.values())
            )),
            tooltip=[
                alt.Tooltip("STATE:N", title="시/도"),
                alt.Tooltip("CITY:N", title="시/군/구"),
                alt.Tooltip("FGI_NORM:Q", title="FGI", format=".1f"),
                alt.Tooltip("CONTRACT_TREND_NORM:Q", title="트렌드", format=".2f"),
                alt.Tooltip("상태:N"),
            ],
        ).properties(height=400)
        st.altair_chart(scatter, width="stretch")
    else:
        st.warning("FGI와 계약 데이터를 조인할 수 없습니다.")

# ──────────────────────────────────────────────
# TAB 4: ASK DATA
# ──────────────────────────────────────────────
with tab_ask:
    st.subheader("💬 데이터에게 질문하기")
    st.caption("질문하면 AI가 데이터를 직접 조회해서 결과를 보여드립니다.")

    example_qs = [
        "FGI는 상승했는데 계약은 감소한 지역은?",
        "변동성이 가장 큰 지역 Top 5는?",
        "서울에서 FGI가 가장 높은 구는?",
        "최근 3개월 계약 증가율이 가장 높은 지역은?",
    ]

    st.markdown("**예시 질문:**")
    ex_cols = st.columns(len(example_qs))
    for i, eq in enumerate(example_qs):
        if ex_cols[i].button(eq, key=f"ex_{i}"):
            st.session_state["ask_input"] = eq

    user_q = st.text_input("질문 입력", value=st.session_state.get("ask_input", ""), placeholder="부동산 시장에 대해 질문하세요...")

    if user_q:
        with st.spinner("AI가 데이터를 분석하고 있습니다..."):
            sql_prompt = f"""당신은 Snowflake SQL 전문가입니다. 아래 테이블 정보를 바탕으로 사용자 질문에 맞는 SQL을 작성하세요.

[테이블 1: PRICE_BASE.PUBLIC.CALC_FGI_INTERNET_NN_MEAN]
- YEAR_MONTH (DATE): 기간 (2023-01 ~ 2025-12)
- STATE (VARCHAR): 시/도 (서울, 경기, 부산 등 17개)
- CITY (VARCHAR): 시/군/구 (172개)
- CONTRACT_COUNT (NUMBER): 거래건수
- MOVEMENT_RATE (NUMBER): 이동률
- MEAN_MEME_PRICE (NUMBER): 평균 매매가 (만원)
- MEAN_JEONSE_PRICE (NUMBER): 평균 전세가 (만원)
- CHANGE_MEME_PRICE_RATE (NUMBER): 매매가 변동률 (%)
- REVERSE_JEONSE_PER_MEME (NUMBER): 역전세율 (%)
- RATE_GAP (NUMBER): 금리갭
- PRICE_MOMENTUM_3M (NUMBER): 3개월 가격 모멘텀
- PRICE_VOLATILITY_6M (FLOAT): 6개월 가격 변동성
- FEAR_GREED_INDEX (FLOAT): 공포탐욕지수

[테이블 2: CONTRACT_TRENDS.PUBLIC.CONTRACT_TREND_NORMALIZED]
- INSTALL_STATE (VARCHAR): 시/도
- INSTALL_CITY (VARCHAR): 시/군/구
- YEAR_MONTH (DATE): 기간 (2023-07 ~ 2026-04)
- CONTRACT_COUNT (NUMBER): 계약건수
- CONTRACT_GROWTH (NUMBER): 계약 증가율
- CONTRACT_TREND_NORM (NUMBER): 계약트렌드 정규화 점수

조인 조건: 테이블1.STATE = 테이블2.INSTALL_STATE AND 테이블1.CITY = 테이블2.INSTALL_CITY AND 테이블1.YEAR_MONTH = 테이블2.YEAR_MONTH

규칙:
1. SELECT 문만 작성 (INSERT/UPDATE/DELETE 금지)
2. 결과를 사람이 읽기 쉽게 컬럼명은 한글 별칭 사용 (예: STATE AS "시도")
3. 숫자는 ROUND 처리
4. 결과는 최대 20행
5. SQL만 출력. 설명, 마크다운, 코드블록 기호 없이 순수 SQL만 작성

질문: {user_q}"""

            try:
                safe_sql_prompt = sql_prompt.replace("'", "''")
                llm_result = session.sql(
                    f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', '{safe_sql_prompt}') AS SQL_QUERY"
                ).to_pandas()
                generated_sql = llm_result["SQL_QUERY"].iloc[0].strip()
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

                if generated_sql.upper().startswith("SELECT"):
                    try:
                        query_result = session.sql(generated_sql).to_pandas()

                        insight_prompt = f"""사용자가 "{user_q}"라고 질문했고, 아래 데이터가 조회되었습니다.
이 결과를 바탕으로 핵심 인사이트를 3~5줄로 요약하세요.
- 구체적인 지역명과 수치를 포함
- 비교/순위를 활용
- 시장 상황에 대한 해석 포함
- 한국어로 작성

데이터:
{query_result.head(15).to_string(index=False)}"""
                        safe_insight = insight_prompt.replace("'", "''")
                        insight_result = session.sql(
                            f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', '{safe_insight}') AS INSIGHT"
                        ).to_pandas()
                        insight = insight_result["INSIGHT"].iloc[0]

                        st.markdown("#### 💡 분석 인사이트")
                        st.markdown(insight)

                        st.markdown("---")
                        st.markdown("#### 📋 조회 결과")
                        st.dataframe(query_result, hide_index=True, width=800)

                        num_cols = query_result.select_dtypes(include=[np.number]).columns.tolist()
                        str_cols = query_result.select_dtypes(include=["object"]).columns.tolist()

                        if len(str_cols) >= 1 and len(num_cols) >= 1:
                            st.markdown("#### 📊 시각화")
                            chart_x = str_cols[0]
                            chart_y = num_cols[0]

                            if len(query_result) <= 30:
                                auto_chart = alt.Chart(query_result).mark_bar(
                                    cornerRadiusTopLeft=4, cornerRadiusTopRight=4
                                ).encode(
                                    x=alt.X(f"{chart_x}:N", title=chart_x, sort="-y"),
                                    y=alt.Y(f"{chart_y}:Q", title=chart_y),
                                    color=alt.Color(f"{chart_y}:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=None),
                                    tooltip=[alt.Tooltip(f"{chart_x}:N")] + [alt.Tooltip(f"{c}:Q", format=",.1f") for c in num_cols],
                                ).properties(height=400)
                                st.altair_chart(auto_chart, width="stretch")

                                if len(num_cols) >= 2:
                                    chart_y2 = num_cols[1]
                                    scatter_ask = alt.Chart(query_result).mark_circle(size=80).encode(
                                        x=alt.X(f"{num_cols[0]}:Q", title=num_cols[0]),
                                        y=alt.Y(f"{num_cols[1]}:Q", title=num_cols[1]),
                                        color=alt.Color(f"{chart_x}:N", legend=None),
                                        tooltip=[alt.Tooltip(f"{chart_x}:N")] + [alt.Tooltip(f"{c}:Q", format=",.1f") for c in num_cols],
                                    ).properties(height=350)
                                    st.altair_chart(scatter_ask, width="stretch")

                        with st.expander("실행된 SQL 보기"):
                            st.code(generated_sql, language="sql")

                    except Exception as exec_err:
                        st.warning("SQL 실행 중 오류가 발생했습니다. 다른 질문을 시도해 보세요.")
                        with st.expander("오류 상세"):
                            st.code(str(exec_err))
                            st.code(generated_sql, language="sql")
                else:
                    st.markdown("#### 분석 결과")
                    st.markdown(generated_sql)
            except Exception as e:
                st.error(f"AI 호출 중 오류가 발생했습니다: {str(e)}")

st.markdown("---")
model_status = "Registry 연동" if registry_available else "테이블 기반"
st.markdown(
    f"<p style='text-align:center;color:gray;font-size:0.8em;'>"
    f"데이터: PRICE_BASE.PUBLIC &amp; CONTRACT_TRENDS.PUBLIC | "
    f"ML 모델: {model_status} (FGI_ANALYSIS_RF, FGI_PREDICT_XGB) | Powered by Snowflake Cortex</p>",
    unsafe_allow_html=True,
)
