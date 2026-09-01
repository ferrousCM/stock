"""DART(전자공시시스템) OpenAPI 연동 — 종목별 공식 공시 목록.

무료 API지만 사용하려면 https://opendart.fss.or.kr 에서 본인이 직접 키를
발급받아야 한다. 키는 두 경로 중 하나로 넣는다:

  - 로컬: 프로젝트 루트 `.env` 파일에 `DART_API_KEY=발급받은키` (.env는 gitignore됨)
  - Streamlit Cloud: 앱 설정 Secrets에 `DART_API_KEY = "발급받은키"` (섹션 없이 최상위)

`config.get_dart_api_key()`가 매 호출마다 환경변수 → Streamlit secrets 순으로 조회한다
(import 시점에 굳히면 secrets 지연 로딩을 못 잡는다). 키가 없으면 DartKeyMissing,
서버 연결에 실패하면 DartUnavailable을 던진다 — 호출부(app.py / pages/모니터링.py)에서
둘 다 잡아 안내 메시지로 보여준다.
"""

from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from . import config
from .cache_utils import cache_path, is_fresh

_BASE = "https://opendart.fss.or.kr/api"
_CORP_CODE_TTL_SEC = 24 * 60 * 60  # 상장사 목록은 하루 캐시로 충분
# (연결, 응답) 타임아웃. opendart.fss.or.kr는 한국 밖 IP(예: Streamlit Cloud, 미국
# 리전)에서 응답이 아주 느리거나 연결 자체가 드롭되는 일이 잦다 — 20초씩 매달리지 않게
# 짧게 잡고, 실패하면 DartUnavailable로 바꿔 던진다.
_HTTP_TIMEOUT = (4, 8)
# 한 번 연결에 실패하면 이 시간 동안은 네트워크를 다시 때리지 않고 즉시 DartUnavailable을
# 던진다. st.tabs가 매 렌더마다 공시 탭 본문을 실행하므로, 이 쿨다운이 없으면 해외 배포에서
# 페이지를 열 때마다 타임아웃(수 초)을 반복해서 문다. 프로세스 재시작 시 초기화된다.
_UNAVAILABLE_COOLDOWN_SEC = 300
_unavailable_until = 0.0


class DartKeyMissing(RuntimeError):
    """DART_API_KEY가 설정되지 않았을 때."""


class DartUnavailable(RuntimeError):
    """DART 서버에 연결하지 못했을 때 (타임아웃·네트워크 오류·HTTP 에러).

    opendart.fss.or.kr는 한국 외 지역에서 접속하면 응답이 없거나 차단되는 일이 잦다 —
    Streamlit Cloud 배포에서 특히 자주 발생한다. 호출부(app.py / pages/모니터링.py)는
    DartKeyMissing과 함께 이 예외도 잡아 안내 메시지로 보여줘야 한다 (안 잡으면 대시보드
    전체가 죽는다 — st.tabs는 활성 탭이 아니어도 모든 탭 본문을 매 렌더 실행한다).
    """


_UNAVAILABLE_MSG = (
    "DART 서버(opendart.fss.or.kr)에 연결하지 못했습니다. 한국 외 지역에서 "
    "접속하면(예: Streamlit Cloud) 응답이 없거나 차단될 수 있습니다."
)


def _dart_get(endpoint: str, params: dict) -> requests.Response:
    """DART API GET 요청. 네트워크/HTTP 오류는 전부 DartUnavailable로 바꿔 던진다.

    직전에 실패한 적이 있으면(쿨다운 중) 네트워크를 때리지 않고 바로 던진다.
    """
    global _unavailable_until
    if time.monotonic() < _unavailable_until:
        raise DartUnavailable(_UNAVAILABLE_MSG + " (직전 실패 후 잠시 재시도를 멈춘 상태)")
    try:
        r = requests.get(f"{_BASE}/{endpoint}", params=params, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        _unavailable_until = time.monotonic() + _UNAVAILABLE_COOLDOWN_SEC
        raise DartUnavailable(_UNAVAILABLE_MSG) from e
    return r


def _get_key() -> str:
    key = config.get_dart_api_key()
    if not key:
        raise DartKeyMissing(
            "DART_API_KEY가 설정되지 않았습니다. https://opendart.fss.or.kr 에서 키를 발급받아 "
            "→ 로컬: 프로젝트 루트 .env 파일에 DART_API_KEY=발급받은키 "
            '→ Streamlit Cloud: 앱 설정 Secrets에 DART_API_KEY = "발급받은키" 를 '
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
    r = _dart_get("corpCode.xml", {"crtfc_key": key})

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
    r = _dart_get(
        "list.json",
        {
            "crtfc_key": key,
            "corp_code": corp_code,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "page_count": 50,
        },
    )
    try:
        j = r.json()
    except ValueError as e:  # 200이지만 JSON이 아님 (차단 안내 페이지 등)
        raise DartUnavailable(_UNAVAILABLE_MSG) from e

    if j.get("status") != "000":  # "013" = 조회된 데이터 없음 등
        df = pd.DataFrame(columns=["rcept_dt", "report_nm", "flr_nm", "url"])
    else:
        df = pd.DataFrame(j.get("list", []))
        df["url"] = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + df["rcept_no"]
        df = df[["rcept_dt", "report_nm", "flr_nm", "url"]].reset_index(drop=True)

    if use_cache:
        df.to_parquet(path)
    return df
