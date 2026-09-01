"""DART(전자공시시스템) OpenAPI 연동 — 종목별 공식 공시 목록.

무료 API지만 사용하려면 https://opendart.fss.or.kr 에서 본인이 직접 키를
발급받아야 한다. 키는 두 경로 중 하나로 넣는다:

  - 로컬: 프로젝트 루트 `.env` 파일에 `DART_API_KEY=발급받은키` (.env는 gitignore됨)
  - Streamlit Cloud: 앱 설정 Secrets에 `DART_API_KEY = "발급받은키"` (섹션 없이 최상위)

`config.get_dart_api_key()`가 매 호출마다 환경변수 → Streamlit secrets 순으로 조회한다
(import 시점에 굳히면 secrets 지연 로딩을 못 잡는다). 키가 없으면 이 모듈의 함수들은
DartKeyMissing을 던진다 — 호출부(app.py)에서 잡아서 안내 메시지로 보여준다.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from . import config
from .cache_utils import cache_path, is_fresh

_BASE = "https://opendart.fss.or.kr/api"
_CORP_CODE_TTL_SEC = 24 * 60 * 60  # 상장사 목록은 하루 캐시로 충분


class DartKeyMissing(RuntimeError):
    """DART_API_KEY가 설정되지 않았을 때."""


def _get_key() -> str:
    key = config.get_dart_api_key()
    if not key:
        raise DartKeyMissing(
            "DART_API_KEY가 설정되지 않았습니다. https://opendart.fss.or.kr 에서 키를 발급받아 "
            "→ 로컬: 프로젝트 루트 .env 파일에 DART_API_KEY=발급받은키 "
            "→ Streamlit Cloud: 앱 설정 Secrets에 DART_API_KEY = \"발급받은키\" 를 "
            "(다른 [섹션] 안이 아니라 파일 최상위에) 추가하세요."
        )
    return key


def _corp_code_map(use_cache: bool = True) -> pd.DataFrame:
    """전체 상장사 corp_code <-> stock_code 매핑. DART는 6자리 종목코드가 아니라
    자체 8자리 corp_code를 쓰므로, 공시 조회 전에 이 매핑이 필요하다.
    """
    path = cache_path("dart_corpcode", "all")
    if use_cache and is_fresh(path, ttl_sec=_CORP_CODE_TTL_SEC):
        return pd.read_parquet(path)

    key = _get_key()
    r = requests.get(f"{_BASE}/corpCode.xml", params={"crtfc_key": key}, timeout=20)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml_bytes)

    rows = [
        {
            "corp_code": item.findtext("corp_code"),
            "corp_name": item.findtext("corp_name"),
            "stock_code": (item.findtext("stock_code") or "").strip(),
        }
        for item in root.findall("list")
    ]
    df = pd.DataFrame(rows)
    df = df[df["stock_code"] != ""].reset_index(drop=True)  # 비상장사는 stock_code가 빈 문자열

    if use_cache:
        df.to_parquet(path)
    return df


def corp_code_for(stock_code: str) -> str | None:
    """6자리 종목코드 -> DART corp_code. 못 찾으면 None."""
    df = _corp_code_map()
    hit = df[df["stock_code"] == stock_code]
    return hit["corp_code"].iloc[0] if not hit.empty else None


def fetch_disclosures(stock_code: str, days_back: int = 90, use_cache: bool = True) -> pd.DataFrame:
    """최근 공시 목록. columns: rcept_dt, report_nm, flr_nm, url

    종목코드를 DART corp_code로 매핑하지 못하면(비상장·최근 상장 등) 빈 DataFrame.
    """
    key = _get_key()  # 여기서 먼저 검증 — 키 없으면 corp_code 조회 전에 바로 에러

    corp_code = corp_code_for(stock_code)
    if not corp_code:
        return pd.DataFrame(columns=["rcept_dt", "report_nm", "flr_nm", "url"])

    path = cache_path("dart_list", f"{stock_code}|{days_back}")
    if use_cache and is_fresh(path, ttl_sec=config.NEWS_CACHE_TTL_SEC):
        return pd.read_parquet(path)

    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=days_back)
    r = requests.get(
        f"{_BASE}/list.json",
        params={
            "crtfc_key": key,
            "corp_code": corp_code,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "page_count": 50,
        },
        timeout=15,
    )
    j = r.json()

    if j.get("status") != "000":  # "013" = 조회된 데이터 없음 등
        df = pd.DataFrame(columns=["rcept_dt", "report_nm", "flr_nm", "url"])
    else:
        df = pd.DataFrame(j.get("list", []))
        df["url"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + df["rcept_no"]
        df = df[["rcept_dt", "report_nm", "flr_nm", "url"]].reset_index(drop=True)

    if use_cache:
        df.to_parquet(path)
    return df
