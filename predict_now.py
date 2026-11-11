# -*- coding: utf-8 -*-
"""
예측 실행 (4단계): 최근 3시간 er_snapshots → LightGBM 60분 뒤 가용 병상 예측 → er_predictions 저장

사용 (저장소 폴더에서):
    시험(저장 안 함, 화면에만 표시):  python predict_now.py
    저장 + 2일 지난 예측 삭제:        python predict_now.py --write

환경 변수
    SUPABASE_URL          예: https://xxxx.supabase.co
    SUPABASE_KEY          시험용 읽기 키(anon 키 가능)
    SUPABASE_SERVICE_KEY  저장용 키(service_role). --write일 때 필수

모델 정보는 model_info.json (모델 파일·이름·검증 MAE). 주간 재학습으로 모델을 바꾸면 이 파일과 models/ 만 교체.
특성 계산은 Safety_Golden/tools/prediction/train_lgbm.py와 반드시 같아야 함.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

KST = "Asia/Seoul"
FFILL_LIMIT = 3          # 학습과 동일: 값이 없는 칸은 직전 값으로 최대 30분
LOOKBACK_MIN = 200       # 120분 전 차이 + 3시간 평균에 필요한 만큼
STALE_LIMIT_MIN = 40     # 최신 수집이 이보다 오래되면 실패로 처리(수집 중단 알림)
KEEP_DAYS = 2            # 목표 시각이 2일 지난 예측은 삭제
ID_WINDOW = 15000        # 최신 id부터 이만큼 거슬러 읽음 (10분에 약 415행 → 약 6시간, 200분보다 넉넉히)
RETRIES = 3              # 조회 실패 시 재시도 횟수
VALS = ["hvec", "hvs01", "hvoc", "hvicc", "hvgc"]
HOLIDAYS = pd.to_datetime(["2026-10-09", "2026-12-25"]).tz_localize(KST)
DEMO_HPID = "A1100021"   # 시연 병원(삼육서울병원) — 시험 모드에서 따로 표시
NO_INFO = "정보 없음"


def er_status(hvec, hvs01):
    """앱 lib/utils/er_status.dart와 같은 규칙 (시험 모드 확인용)"""
    hv = np.asarray(hvec, dtype=float)
    hs = np.asarray(hvs01, dtype=float)
    out = np.full(hv.shape, NO_INFO, dtype=object)
    valid = ~np.isnan(hv)
    over = valid & (hv < 0)
    ok = valid & ~over & (hs > 0) & (hv <= hs)
    rate = np.where(ok, hv / np.where(hs > 0, hs, 1) * 100, np.nan)
    out[ok & (rate >= 80)] = "여유"
    out[ok & (rate >= 50) & (rate < 80)] = "보통"
    out[ok & (rate < 50)] = "혼잡"
    out[over] = "과밀"
    return out


def headers(key, extra=None):
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    if extra:
        h.update(extra)
    return h


def check(r, what):
    if not r.ok:
        sys.exit(f"{what} 실패: HTTP {r.status_code}\n{r.text[:500]}")


def get_json(url, key, params, what):
    """일시적인 DB 시간 초과(HTTP 500, 57014)에 대비해 15초 간격으로 최대 3번 시도"""
    for attempt in range(1, RETRIES + 1):
        r = requests.get(f"{url}/rest/v1/er_snapshots", params=params, headers=headers(key), timeout=30)
        if r.ok:
            return r.json()
        print(f"  {what} 실패 {attempt}/{RETRIES}: HTTP {r.status_code} {r.text[:120]}")
        if attempt < RETRIES:
            time.sleep(15)
    check(r, what)


def fetch_snapshots(url, key):
    """기본키(id) 범위로만 읽음 (DB 변경 없음).
    - recorded_at 조건: recorded_at 단독 인덱스가 없어 표 전체를 훑다 시간 초과(11/11)
    - id 내림차순 + offset: 대부분 되지만 가끔 시간 초과(11/12 01시 실패 메일)
    → 최신 id 1개를 먼저 찾고, 그보다 ID_WINDOW 작은 id부터 'id > 커서' 조건으로 1000행씩 읽음(항상 기본키 범위 조회).
      기간(LOOKBACK_MIN)은 Python에서 거름"""
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=LOOKBACK_MIN)
    top = get_json(url, key, {"select": "id", "order": "id.desc", "limit": 1}, "최신 id 조회")
    if not top:
        sys.exit("er_snapshots가 비어 있습니다")
    cursor = top[0]["id"] - ID_WINDOW
    rows = []
    while True:
        batch = get_json(url, key, {
            "select": "id,hpid,recorded_at," + ",".join(VALS),
            "id": f"gt.{cursor}",
            "order": "id.asc",
            "limit": 1000,
        }, "er_snapshots 조회")
        rows += batch
        if len(batch) < 1000:
            break
        cursor = batch[-1]["id"]
    df = pd.DataFrame(rows)
    df = df[pd.to_datetime(df["recorded_at"], utc=True, format="mixed") >= since]
    if df.empty:
        sys.exit("최근 데이터가 없습니다 (수집 중단 또는 키 권한 확인)")
    print(f"er_snapshots {len(rows):,}행 받음 → 최근 {LOOKBACK_MIN}분 {len(df):,}행 사용")
    return df.drop(columns="id")


def build_features(df):
    """train_lgbm.py의 make_features와 같은 계산을 최근 구간에만 적용"""
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, format="mixed")
    for c in VALS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["slot"] = (df["recorded_at"].dt.tz_convert(KST).dt.floor("10min")
                  .astype("datetime64[ns, Asia/Seoul]"))
    base = df["slot"].max()
    df = df.sort_values("recorded_at").drop_duplicates(["hpid", "slot"], keep="last")

    slots = pd.date_range(base - pd.Timedelta(minutes=LOOKBACK_MIN - 10), base, freq="10min", unit="ns")
    full = pd.MultiIndex.from_product([np.sort(df["hpid"].unique()), slots], names=["hpid", "slot"])
    g = df.set_index(["hpid", "slot"])[VALS].reindex(full)
    g[VALS] = g.groupby(level="hpid")[VALS].ffill(limit=FFILL_LIMIT)
    g = g.reset_index()

    by = g.groupby("hpid", sort=False)["hvec"]
    for k in (1, 2, 3, 6, 12):
        g[f"diff_{k * 10}"] = g["hvec"] - by.shift(k)
    g["mean_60"] = by.rolling(6, min_periods=3).mean().reset_index(level=0, drop=True)
    g["mean_180"] = by.rolling(18, min_periods=6).mean().reset_index(level=0, drop=True)
    g["rate"] = g["hvec"] / g["hvs01"].where(g["hvs01"] > 0)
    g["over"] = (g["hvec"] < 0).astype("int8")
    g["tod"] = g["slot"].dt.hour * 6 + g["slot"].dt.minute // 10
    g["hour"] = g["tod"] // 6
    g["dow"] = g["slot"].dt.day_of_week
    g["off_day"] = ((g["dow"] >= 5) | g["slot"].dt.normalize().isin(HOLIDAYS)).astype("int8")

    now = g[(g["slot"] == base) & g["hvec"].notna()].copy()  # 30분 넘게 갱신 없는 병원은 제외
    return now, base


def predict(now, base, info):
    model = lgb.Booster(model_file=info["file"])
    feats = model.feature_name()
    missing = [f for f in feats if f not in now.columns]
    if missing:
        sys.exit(f"모델이 요구하는 특성 {missing}을 계산하지 않습니다 → 모델 파일 버전 확인 "
                 f"(train_lgbm.py 기본값으로 다시 학습한 파일인지)")
    X = now[feats].copy()
    for c in feats:
        X[c] = X[c].astype("category") if c == "hpid" else X[c].astype("float32")
    pred = now["hvec"].to_numpy() + model.predict(X)  # 모델은 변화량을 예측
    cap = now["hvs01"].where(now["hvs01"] > 0).to_numpy()
    pred = np.where(~np.isnan(cap) & (pred > cap), cap, pred)  # 기준 병상 수 상한 (앱과 동일)

    base_utc = base.tz_convert("UTC")
    target_utc = base_utc + pd.Timedelta(minutes=info["horizon_min"])
    records = []
    for hpid, p, hs in zip(now["hpid"], pred, now["hvs01"]):
        records.append({
            "hpid": str(hpid),
            "base_time": base_utc.isoformat(),
            "horizon_min": info["horizon_min"],
            "target_time": target_utc.isoformat(),
            "pred_hvec": round(float(p), 2),
            "hvs01": None if pd.isna(hs) else int(hs),
            "model": info["model"],
            "model_mae": info["mae"],
        })
    return records, pred


def upsert(url, key, records):
    extra = {"Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"}
    for i in range(0, len(records), 500):
        r = requests.post(f"{url}/rest/v1/er_predictions", headers=headers(key, extra),
                          data=json.dumps(records[i:i + 500]), timeout=30)
        check(r, "er_predictions 저장")


def cleanup(url, key):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.delete(f"{url}/rest/v1/er_predictions", params={"target_time": f"lt.{cutoff}"},
                        headers=headers(key, {"Prefer": "return=minimal"}), timeout=30)
    check(r, "오래된 예측 삭제")


def main():
    write = "--write" in sys.argv
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = (os.environ.get("SUPABASE_SERVICE_KEY") if write
           else os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY"))
    if not url or not key:
        sys.exit("환경 변수 SUPABASE_URL과 키(시험: SUPABASE_KEY / 저장: SUPABASE_SERVICE_KEY)를 설정하세요")

    with open("model_info.json", encoding="utf-8") as f:
        info = json.load(f)

    df = fetch_snapshots(url, key)
    now, base = build_features(df)
    age = (pd.Timestamp.now(tz=KST) - base).total_seconds() / 60
    print(f"기준 시각(KST) {base:%m/%d %H:%M} ({age:.0f}분 전 수집), 예측 병원 {len(now)}곳, 모델 {info['model']}")
    if age > STALE_LIMIT_MIN:
        sys.exit(f"최신 수집이 {age:.0f}분 전입니다 → 수집기(pg_cron) 상태 확인 필요. 예측 저장 안 함")

    records, pred = predict(now, base, info)

    s_now = er_status(now["hvec"], now["hvs01"])
    s_pred = er_status(pred, now["hvs01"])
    print("현재 상태  :", pd.Series(s_now).value_counts().to_dict())
    print("1시간 뒤   :", pd.Series(s_pred).value_counts().to_dict())
    print(f"상태가 바뀔 것으로 예측된 병원: {(s_now != s_pred).sum()}곳")
    demo = now["hpid"].astype(str).to_numpy() == DEMO_HPID
    if demo.any():
        i = int(np.argmax(demo))
        print(f"시연 병원 {DEMO_HPID}: 지금 {now['hvec'].iloc[i]:.0f}/{now['hvs01'].iloc[i]:.0f} ({s_now[i]}) "
              f"→ 1시간 뒤 {pred[i]:.1f} ({s_pred[i]})")

    if not write:
        print("시험 모드: 저장하지 않았습니다 (저장은 --write)")
        return
    upsert(url, key, records)
    cleanup(url, key)
    print(f"저장 완료: er_predictions {len(records)}행, 목표 시각 {KEEP_DAYS}일 지난 예측 삭제")


if __name__ == "__main__":
    main()
