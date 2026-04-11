"""
부동산 거래량 예측 웹 서비스
- Snowflake Registry 모델(XGBOOST_CONTRACT_PREDICTION)을 불러와서 실시간 예측
- Snowflake 테이블에서 데이터 조회

실행: uvicorn main:app --reload
접속: http://localhost:8000
"""

import os
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from snowflake.snowpark import Session
from snowflake.ml.registry import Registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT", "bshthag-jl97537"),
    "user": os.getenv("SNOWFLAKE_USER", "SANGGYOON"),
    "password": os.getenv("SNOWFLAKE_PASSWORD", ""),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "MY_DB"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
    "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
}

MODEL_NAME = "XGBOOST_CONTRACT_PREDICTION"
MODEL_VERSION = "V1_WAREHOUSE"

FEATURES_6M = [
    "FGI_LAG1", "FGI_LAG2", "FGI_LAG3", "FGI_LAG6",
    "MONTH_SIN", "MONTH_COS",
    "CONTRACT_COUNT_LAG1", "CONTRACT_COUNT_LAG2", "CONTRACT_COUNT_LAG3", "CONTRACT_COUNT_LAG6",
    "CHANGE_MEME_PRICE_RATE_LAG1", "CHANGE_MEME_PRICE_RATE_LAG2",
    "CHANGE_MEME_PRICE_RATE_LAG3", "CHANGE_MEME_PRICE_RATE_LAG6",
]

_TABLE_SUFFIXES = [
    "CALC_FGI",
    "CALC_FGI_TEST",
    "CALC_FGI_TEST_NN_REMOVE",
    "CALC_FGI_TRAINING",
    "CALC_FGI_TRAINING_NN_MEAN",
    "CALC_FGI_TRAINING_NN_REMOVE",
    "CALC_FGI_TRAINING_NN_ZERO",
    "FEATURE_NN_MEAN",
    "FEATURE_NN_MEAN_3M",
    "FEATURE_NN_MEAN_6M",
    "FEATURE_NN_REMOVE",
    "FEATURE_NN_REMOVE_3M",
    "FEATURE_NN_REMOVE_6M",
    "FEATURE_NN_ZERO",
    "FEATURE_NN_ZERO_3M",
    "FEATURE_NN_ZERO_6M",
    "FEATURE_TEST",
]

TABLES: dict[str, str] = {}
for _prefix in ("RENTAL", "INTERNET"):
    for _suffix in _TABLE_SUFFIXES:
        _name = f"{_prefix}_{_suffix}"
        TABLES[_name] = f"MY_DB.PUBLIC.{_name}"

sf_session: Optional[Session] = None
model_cache: dict = {}  # {(name, version): model_version_obj}


def get_session() -> Session:
    global sf_session
    if sf_session is not None:
        try:
            sf_session.sql("SELECT 1").collect()
            return sf_session
        except Exception:
            logger.warning("기존 세션 만료. 재연결 시도.")
            sf_session = None

    if not SNOWFLAKE_CONFIG["password"]:
        raise RuntimeError(
            "SNOWFLAKE_PASSWORD 환경변수가 설정되지 않았습니다.\n"
            "실행 예시: SNOWFLAKE_PASSWORD=yourpass uvicorn main:app --reload"
        )

    try:
        sf_session = Session.builder.configs(SNOWFLAKE_CONFIG).create()
        logger.info("Snowflake 세션 생성 완료.")
        return sf_session
    except Exception as e:
        logger.error(f"Snowflake 연결 실패: {e}")
        raise RuntimeError(f"Snowflake 연결 실패: {e}")


def get_model(name: str = MODEL_NAME, version: str = MODEL_VERSION):
    key = (name, version)
    if key in model_cache:
        return model_cache[key]

    session = get_session()
    try:
        reg = Registry(session=session, database_name="MY_DB", schema_name="PUBLIC")
        mv = reg.get_model(name).version(version)
        model_cache[key] = mv
        logger.info(f"모델 로드 완료: {name} / {version}")
        return mv
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise RuntimeError(f"모델 로드 실패: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_session()
        get_model(MODEL_NAME, MODEL_VERSION)
        logger.info("서버 초기화 완료: Snowflake 연결 + 모델 로드 성공")
    except Exception as e:
        logger.error(f"서버 초기화 실패: {e}")
        logger.error("서버는 시작되지만, API 호출 시 재연결을 시도합니다.")
    yield
    global sf_session, model_cache
    model_cache.clear()
    if sf_session is not None:
        try:
            sf_session.close()
            logger.info("Snowflake 세션 종료.")
        except Exception:
            pass


app = FastAPI(title="부동산 거래량 예측 API", lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


class PredictRequest(BaseModel):
    table: str = "FEATURE_TEST"
    state: Optional[str] = None
    city: Optional[str] = None
    limit: int = 200
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION


def safe_query(sql: str) -> pd.DataFrame:
    session = get_session()
    try:
        return session.sql(sql).to_pandas()
    except Exception as e:
        logger.error(f"쿼리 실패: {sql[:100]}... -> {e}")
        raise HTTPException(status_code=500, detail=f"쿼리 실패: {e}")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def list_models():
    try:
        session = get_session()
        model_rows = session.sql("SHOW MODELS IN SCHEMA MY_DB.PUBLIC").collect()
        result = []
        for row in model_rows:
            name = row["name"]
            try:
                ver_rows = session.sql(f"SHOW VERSIONS IN MODEL MY_DB.PUBLIC.{name}").collect()
                versions = [v["name"] for v in ver_rows]
            except Exception:
                versions = []
            result.append({"name": name, "versions": versions})
        return {"models": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tables")
async def list_tables():
    grouped: dict[str, list[str]] = {}
    feature_grouped: dict[str, list[str]] = {}
    for name in TABLES:
        prefix = name.split("_")[0]
        grouped.setdefault(prefix, []).append(name)
        if "FEATURE_" in name:
            feature_grouped.setdefault(prefix, []).append(name)
    return {
        "tables": list(TABLES.keys()),
        "grouped": grouped,
        "feature_grouped": feature_grouped,
    }


@app.get("/api/filters/{table_name}")
async def get_filters(table_name: str):
    if table_name not in TABLES:
        raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다.")

    full_name = TABLES[table_name]
    try:
        states_df = safe_query(f"SELECT DISTINCT STATE FROM {full_name} ORDER BY STATE")
        states = states_df["STATE"].tolist()

        cities_df = safe_query(f"SELECT DISTINCT STATE, CITY FROM {full_name} ORDER BY STATE, CITY")
        cities_by_state = {}
        for _, row in cities_df.iterrows():
            s = row["STATE"]
            c = row["CITY"]
            if s not in cities_by_state:
                cities_by_state[s] = []
            cities_by_state[s].append(c)

        return {"states": states, "cities_by_state": cities_by_state}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
async def predict(req: PredictRequest):
    if req.table not in TABLES:
        raise HTTPException(status_code=404, detail=f"테이블 '{req.table}'을 찾을 수 없습니다.")
    if req.limit < 1 or req.limit > 5000:
        raise HTTPException(status_code=400, detail="limit은 1~5000 사이여야 합니다.")

    full_name = TABLES[req.table]

    where_clauses = []
    if req.state:
        where_clauses.append(f"STATE = '{req.state}'")
    if req.city:
        where_clauses.append(f"CITY = '{req.city}'")
    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    try:
        session = get_session()
        input_df = session.sql(f"SELECT * FROM {full_name}{where_sql} LIMIT {req.limit}")

        mv = get_model(req.model_name, req.model_version)
        result_sp = mv.run(input_df, function_name="predict")
        result = result_sp.to_pandas()

        if result.empty:
            return {
                "status": "empty",
                "message": "조건에 맞는 데이터가 없습니다.",
                "data": [],
                "metrics": None,
                "chart_data": None,
            }

        input_cols = [c.upper() for c in FEATURES_6M]
        extra_cols = ["YEAR_MONTH", "STATE", "CITY", "CONTRACT_COUNT"]
        pred_candidates = [c for c in result.columns if c.upper() not in [x.upper() for x in input_cols + extra_cols + list(result.columns[:0])]]
        if pred_candidates:
            pred_col = pred_candidates[-1]
        else:
            pred_col = result.columns[-1]

        result = result.rename(columns={pred_col: "PREDICTION"})

        for col in result.select_dtypes(include=["number"]).columns:
            result[col] = result[col].round(4)

        metrics = None
        if "CONTRACT_COUNT" in result.columns and "PREDICTION" in result.columns:
            actual = result["CONTRACT_COUNT"].astype(float)
            predicted = result["PREDICTION"].astype(float)
            valid = actual.notna() & predicted.notna()
            actual = actual[valid]
            predicted = predicted[valid]

            if len(actual) > 0:
                diff = actual.values - predicted.values
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                ss_res = float(np.sum(diff ** 2))
                ss_tot = float(np.sum((actual.values - np.mean(actual.values)) ** 2))
                r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
                metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "count": len(actual)}

        chart_data = None
        if "CONTRACT_COUNT" in result.columns and "PREDICTION" in result.columns:
            chart_df = result[["CONTRACT_COUNT", "PREDICTION"]].dropna()
            chart_data = {
                "actual": chart_df["CONTRACT_COUNT"].tolist(),
                "predicted": chart_df["PREDICTION"].tolist(),
            }

            if "YEAR_MONTH" in result.columns:
                monthly = result.groupby("YEAR_MONTH").agg(
                    actual_avg=("CONTRACT_COUNT", "mean"),
                    predicted_avg=("PREDICTION", "mean"),
                ).reset_index()
                monthly = monthly.sort_values("YEAR_MONTH")
                chart_data["monthly"] = {
                    "labels": [str(d)[:10] for d in monthly["YEAR_MONTH"].tolist()],
                    "actual_avg": [round(v, 2) for v in monthly["actual_avg"].tolist()],
                    "predicted_avg": [round(v, 2) for v in monthly["predicted_avg"].tolist()],
                }

            if "STATE" in result.columns:
                by_state = result.groupby("STATE").agg(
                    actual_avg=("CONTRACT_COUNT", "mean"),
                    predicted_avg=("PREDICTION", "mean"),
                    count=("CONTRACT_COUNT", "count"),
                ).reset_index().sort_values("count", ascending=False)
                chart_data["by_state"] = {
                    "labels": by_state["STATE"].tolist(),
                    "actual_avg": [round(v, 2) for v in by_state["actual_avg"].tolist()],
                    "predicted_avg": [round(v, 2) for v in by_state["predicted_avg"].tolist()],
                }

        display_cols = ["YEAR_MONTH", "STATE", "CITY", "CONTRACT_COUNT", "PREDICTION"]
        display_cols = [c for c in display_cols if c in result.columns]
        table_data = result[display_cols].fillna("").to_dict(orient="records")

        return {
            "status": "success",
            "metrics": metrics,
            "chart_data": chart_data,
            "data": table_data,
            "total_rows": len(result),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")


def _resolve_pred_col(result: pd.DataFrame) -> pd.DataFrame:
    input_cols_upper = [c.upper() for c in FEATURES_6M]
    extra_upper = {"YEAR_MONTH", "STATE", "CITY", "CONTRACT_COUNT"}
    pred_candidates = [
        c for c in result.columns
        if c.upper() not in input_cols_upper and c.upper() not in extra_upper
    ]
    pred_col = pred_candidates[-1] if pred_candidates else result.columns[-1]
    return result.rename(columns={pred_col: "PREDICTION"})


@app.post("/api/insights")
async def get_insights(req: PredictRequest):
    if req.table not in TABLES:
        raise HTTPException(status_code=404, detail=f"테이블 '{req.table}'을 찾을 수 없습니다.")

    full_name = TABLES[req.table]
    try:
        session = get_session()
        input_df = session.sql(f"SELECT * FROM {full_name} LIMIT 5000")
        mv = get_model(req.model_name, req.model_version)
        result = mv.run(input_df, function_name="predict").to_pandas()

        if result.empty:
            return {"status": "empty"}

        result = _resolve_pred_col(result)

        cols_upper = {c.upper(): c for c in result.columns}

        def col(name):
            return cols_upper.get(name.upper())

        # 1. FGI by STATE
        fgi_by_state = None
        fgi_c = col("FGI_LAG1")
        state_c = col("STATE")
        if fgi_c and state_c:
            df = result.groupby(state_c)[fgi_c].mean().reset_index().sort_values(fgi_c, ascending=False)
            fgi_by_state = {
                "labels": df[state_c].tolist(),
                "values": [round(float(v), 2) if not (np.isnan(v) or np.isinf(v)) else 0 for v in df[fgi_c]],
            }

        # 2. 집값 변화율 상위/하위 10 by CITY
        price_top10_up = price_top10_down = None
        price_c = col("CHANGE_MEME_PRICE_RATE_LAG1")
        city_c = col("CITY")
        if price_c and state_c and city_c:
            df = result.groupby([state_c, city_c])[price_c].mean().reset_index()
            df = df.dropna(subset=[price_c])
            df["label"] = df[state_c] + " " + df[city_c]

            top_up = df.nlargest(10, price_c)
            price_top10_up = {
                "labels": top_up["label"].tolist(),
                "values": [round(float(v), 4) for v in top_up[price_c]],
            }
            top_down = df.nsmallest(10, price_c)
            price_top10_down = {
                "labels": top_down["label"].tolist(),
                "values": [round(float(v), 4) for v in top_down[price_c]],
            }

        # 3. CONTRACT_COUNT 예상 변화율 상위/하위 10 by CITY
        contract_top10_up = contract_top10_down = None
        lag1_c = col("CONTRACT_COUNT_LAG1")
        pred_c = "PREDICTION"
        if lag1_c and pred_c in result.columns and city_c and state_c:
            result["_CHANGE_RATE"] = (
                (result[pred_c] - result[lag1_c]) / (result[lag1_c].abs() + 1e-6) * 100
            )
            df = result.groupby([state_c, city_c])["_CHANGE_RATE"].mean().reset_index()
            df = df.dropna(subset=["_CHANGE_RATE"])
            df["label"] = df[state_c] + " " + df[city_c]

            top_up = df.nlargest(10, "_CHANGE_RATE")
            contract_top10_up = {
                "labels": top_up["label"].tolist(),
                "values": [round(float(v), 2) for v in top_up["_CHANGE_RATE"]],
            }
            top_down = df.nsmallest(10, "_CHANGE_RATE")
            contract_top10_down = {
                "labels": top_down["label"].tolist(),
                "values": [round(float(v), 2) for v in top_down["_CHANGE_RATE"]],
            }

        return {
            "status": "success",
            "fgi_by_state": fgi_by_state,
            "price_top10_up": price_top10_up,
            "price_top10_down": price_top10_down,
            "contract_top10_up": contract_top10_up,
            "contract_top10_down": contract_top10_down,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인사이트 분석 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인사이트 분석 실패: {e}")


@app.get("/api/data/{table_name}")
async def get_data(
    table_name: str,
    limit: int = Query(default=100, ge=1, le=1000),
    state: Optional[str] = Query(default=None),
):
    if table_name not in TABLES:
        raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다.")

    full_name = TABLES[table_name]
    where = f" WHERE STATE = '{state}'" if state else ""
    try:
        df = safe_query(f"SELECT * FROM {full_name}{where} LIMIT {limit}")

        stats = None
        num_cols = df.select_dtypes(include="number")
        if not num_cols.empty:
            stats = {}
            for col in num_cols.columns:
                def safe_float(v):
                    f = float(v)
                    return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
                stats[col] = {
                    "mean": safe_float(num_cols[col].mean()),
                    "std": safe_float(num_cols[col].std()),
                    "min": safe_float(num_cols[col].min()),
                    "max": safe_float(num_cols[col].max()),
                }

        count_df = safe_query(f"SELECT COUNT(*) AS CNT FROM {full_name}{where}")
        total_count = int(count_df["CNT"].iloc[0])

        return {
            "table": table_name,
            "total_rows": total_count,
            "returned_rows": len(df),
            "columns": list(df.columns),
            "data": df.fillna("").to_dict(orient="records"),
            "stats": stats,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    status = {"server": "running", "snowflake": "disconnected", "model": "not loaded"}
    try:
        get_session()
        status["snowflake"] = "connected"
    except Exception as e:
        status["snowflake_error"] = str(e)

    if model_cache:
        loaded = [f"{n}/{v}" for n, v in model_cache]
        status["model"] = ", ".join(loaded)

    return status


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  부동산 거래량 예측 웹 서비스")
    print("  http://localhost:8000")
    print("=" * 60)
    print("\n[필수] 환경변수 설정:")
    print("  SNOWFLAKE_PASSWORD=비밀번호")
    print("\n[선택] 환경변수:")
    print("  SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE 등")
    print("=" * 60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
