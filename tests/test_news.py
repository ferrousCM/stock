"""뉴스/감성/공시 모듈 테스트.

sentiment는 순수 함수라 오프라인. news/dart는 외부 사이트·API를 치므로 network 마커.
dart는 DART_API_KEY가 없으면 자동으로 skip한다 (키 발급은 사용자 몫).
"""

import sys
import types

import pandas as pd
import pytest

from src import config, dart, news, sentiment

# ----------------------------------------------------------- sentiment (오프라인)


def test_sentiment_positive():
    result = sentiment.score("실적 개선에 목표주가 상향, 강세 지속 기대감")
    assert result["label"] == "긍정"
    assert result["score"] > 0


def test_sentiment_negative():
    result = sentiment.score("실적 부진에 목표주가 하향, 급락 우려 확산")
    assert result["label"] == "부정"
    assert result["score"] < 0


def test_sentiment_neutral_on_no_keywords():
    result = sentiment.score("오늘 회사는 정기 이사회를 개최했다")
    assert result["label"] == "중립"
    assert result["score"] == 0


# ----------------------------------------------------------- 감성 히스토리 로그 (오프라인)
# data/raw/news_sentiment는 재생성 불가능한 유일한 데이터이므로 테스트는 tmp_path로 격리한다.


def test_log_daily_sentiment_and_history_full(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    code = "TESTCODE"
    news.log_daily_sentiment(
        code, 0.5, positive_count=3, negative_count=1, article_count=4, date=pd.Timestamp("2026-01-01")
    )
    news.log_daily_sentiment(
        code, -0.2, positive_count=1, negative_count=2, article_count=3, date=pd.Timestamp("2026-01-02")
    )

    hist = news.sentiment_history(code)
    assert list(hist.values) == [0.5, -0.2]

    full = news.sentiment_history_full(code)
    assert list(full["positive_count"]) == [3, 1]
    assert list(full["negative_count"]) == [1, 2]
    assert list(full["article_count"]) == [4, 3]


def test_sentiment_history_full_empty_when_no_log(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    full = news.sentiment_history_full("NOPE")
    assert full.empty
    assert list(full.columns) == ["avg_sentiment", "positive_count", "negative_count", "article_count"]


def test_log_sentiment_from_news_counts_labels(tmp_path, monkeypatch):
    """뉴스 DataFrame에서 긍정/부정 기사 수까지 채워 기록해야 한다."""
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    news_df = pd.DataFrame(
        {
            "sentiment_label": ["긍정", "긍정", "부정", "중립"],
            "sentiment_score": [2.0, 1.0, -1.0, 0.0],
        }
    )
    news.log_sentiment_from_news("TESTCODE3", news_df, date=pd.Timestamp("2026-01-01"))

    full = news.sentiment_history_full("TESTCODE3")
    assert full["positive_count"].iloc[0] == 2
    assert full["negative_count"].iloc[0] == 1
    assert full["article_count"].iloc[0] == 4
    assert full["avg_sentiment"].iloc[0] == pytest.approx(0.5)


def test_log_sentiment_from_news_ignores_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    news.log_sentiment_from_news("TESTCODE4", pd.DataFrame())
    assert news.sentiment_history_full("TESTCODE4").empty


def test_sentiment_history_full_coerces_old_none_counts(tmp_path, monkeypatch):
    """카운트가 전부 None인 옛 로그도 숫자(float)로 돌려줘야 학습 피처가 object로 새지 않는다."""
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    news.log_daily_sentiment("OLDCODE", 0.9, date=pd.Timestamp("2026-01-01"))  # 카운트 미지정

    full = news.sentiment_history_full("OLDCODE")
    for c in ["positive_count", "negative_count", "article_count"]:
        assert full[c].dtype.kind == "f", f"{c}가 float이 아님: {full[c].dtype}"
        assert full[c].fillna(0.0).dtype.kind == "f"


def test_log_daily_sentiment_skips_rewrite_when_unchanged(tmp_path, monkeypatch):
    """같은 값을 다시 기록하면 파일을 건드리지 않아야 한다.

    parquet은 같은 내용을 다시 써도 바이트가 달라진다. 이 디렉토리는 git으로 추적하므로
    그냥 덮어쓰면 앱을 켤 때마다 무의미한 변경으로 잡힌다 — 실제로 그런 커밋이 있었다.
    """
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    code, d = "SAMECODE", pd.Timestamp("2026-01-01")
    kwargs = dict(positive_count=6, negative_count=1, article_count=10, date=d)

    news.log_daily_sentiment(code, 0.9, **kwargs)
    path = tmp_path / f"{code}.parquet"
    first = path.read_bytes()

    news.log_daily_sentiment(code, 0.9, **kwargs)  # 완전히 동일한 값
    assert path.read_bytes() == first, "값이 같은데 파일이 재작성됨"

    # 값이 실제로 달라지면 당연히 갱신돼야 한다
    news.log_daily_sentiment(code, 0.4, **kwargs)
    assert path.read_bytes() != first
    assert news.sentiment_history(code).iloc[0] == pytest.approx(0.4)


def test_log_daily_sentiment_rewrites_when_counts_filled_in(tmp_path, monkeypatch):
    """카운트가 None이던 옛 기록에 실제 값이 들어오면 갱신돼야 한다 (건너뛰면 안 됨)."""
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    code, d = "UPGRADE", pd.Timestamp("2026-01-01")

    news.log_daily_sentiment(code, 0.9, date=d)  # 카운트 없이 기록 (옛 방식)
    news.log_daily_sentiment(code, 0.9, positive_count=6, negative_count=1, article_count=10, date=d)

    full = news.sentiment_history_full(code)
    assert full["positive_count"].iloc[0] == 6
    assert full["article_count"].iloc[0] == 10


def test_log_daily_sentiment_appends_new_date(tmp_path, monkeypatch):
    """날짜가 다르면 건너뛰지 않고 행이 늘어야 한다."""
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    code = "MULTIDAY"
    news.log_daily_sentiment(code, 0.5, date=pd.Timestamp("2026-01-01"))
    news.log_daily_sentiment(code, 0.5, date=pd.Timestamp("2026-01-02"))  # 값은 같고 날짜만 다름
    assert len(news.sentiment_history(code)) == 2


def test_log_daily_sentiment_overwrites_same_date(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "_SENTIMENT_LOG_DIR", tmp_path)
    code = "TESTCODE2"
    d = pd.Timestamp("2026-01-01")
    news.log_daily_sentiment(code, 0.1, date=d)
    news.log_daily_sentiment(code, 0.9, date=d)
    hist = news.sentiment_history(code)
    assert len(hist) == 1
    assert hist.iloc[0] == pytest.approx(0.9)


def test_summarize_truncates_and_splits_sentences():
    body = "첫 문장입니다. 두 번째 문장입니다. 세 번째는 안 보여야 합니다."
    summary = news.summarize(body, max_sentences=2, max_chars=200)
    assert "첫 문장" in summary
    assert "두 번째" in summary
    assert "세 번째" not in summary


def test_summarize_empty_body():
    assert news.summarize("") == ""


# --------------------------------------------- enrich_with_sentiment / fetch_news_list_n (오프라인)


def test_enrich_with_sentiment_fills_columns(monkeypatch):
    monkeypatch.setattr(news, "fetch_article_body", lambda o, a, use_cache=True: "호재가 가득한 본문")
    monkeypatch.setattr(news.sentiment, "score", lambda text: {"label": "긍정", "score": 2})
    listing = pd.DataFrame(
        {
            "article_id": ["1", "2"],
            "office_id": ["9", "9"],
            "title": ["제목1", "제목2"],
            "press": ["A", "B"],
            "date": ["", ""],
            "url": ["u1", "u2"],
        }
    )
    out = news.enrich_with_sentiment(listing)
    assert list(out["summary"]) == ["호재가 가득한 본문", "호재가 가득한 본문"]
    assert list(out["sentiment_label"]) == ["긍정", "긍정"]
    assert list(out["sentiment_score"]) == [2, 2]
    # 원본은 그대로 (copy 후 채움)
    assert "summary" not in listing.columns


def test_enrich_with_sentiment_passthrough_empty():
    empty = pd.DataFrame(columns=["article_id", "office_id", "title"])
    assert news.enrich_with_sentiment(empty).empty
    assert news.enrich_with_sentiment(None) is None


def test_fetch_news_list_n_requests_enough_pages(monkeypatch):
    seen = {}

    def _fake_list(code, pages=1, use_cache=True):
        seen["pages"] = pages
        return pd.DataFrame({"article_id": [str(i) for i in range(pages * 20)]})

    monkeypatch.setattr(news, "fetch_news_list", _fake_list)

    out = news.fetch_news_list_n("005930", n=100)
    assert seen["pages"] == 5  # 100 / 20
    assert len(out) == 100

    news.fetch_news_list_n("005930", n=10)
    assert seen["pages"] == 1


# ----------------------------------------------------------- news (네트워크)


@pytest.mark.network
def test_fetch_news_list_has_expected_columns():
    df = news.fetch_news_list("005930", use_cache=False)
    assert not df.empty
    assert {"title", "press", "date", "url", "article_id"} <= set(df.columns)
    assert df["article_id"].is_unique  # 중복 게재 기사 제거 확인


@pytest.mark.network
def test_fetch_news_with_sentiment_end_to_end():
    df = news.fetch_news_with_sentiment("005930", n=3, use_cache=False)
    assert len(df) <= 3
    assert set(df["sentiment_label"]) <= {"긍정", "중립", "부정"}
    assert (df["summary"].str.len() > 0).all()


# ----------------------------------------------------------- dart (네트워크 + 키 필요)


def test_dart_raises_without_key(monkeypatch):
    monkeypatch.setattr(dart.config, "get_dart_api_key", lambda: "")
    with pytest.raises(dart.DartKeyMissing):
        dart.fetch_disclosures("005930")


@pytest.fixture
def _reset_dart_cooldown():
    dart._unavailable_until = 0.0
    yield
    dart._unavailable_until = 0.0


def test_dart_get_converts_network_error_to_unavailable(monkeypatch, _reset_dart_cooldown):
    import requests

    def _boom(*a, **k):
        raise requests.ConnectTimeout("connection timed out")

    monkeypatch.setattr(requests, "get", _boom)
    with pytest.raises(dart.DartUnavailable):
        dart._dart_get("list.json", {"x": 1})


def test_dart_get_cooldown_skips_network_after_failure(monkeypatch, _reset_dart_cooldown):
    import requests

    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise requests.ConnectTimeout("connection timed out")

    monkeypatch.setattr(requests, "get", _boom)

    with pytest.raises(dart.DartUnavailable):
        dart._dart_get("list.json", {"x": 1})
    with pytest.raises(dart.DartUnavailable):
        dart._dart_get("list.json", {"x": 1})  # 쿨다운 중 — 네트워크를 다시 안 친다

    assert calls["n"] == 1


def test_get_dart_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "  from-env  ")
    assert config.get_dart_api_key() == "from-env"  # 양끝 공백 제거


def test_get_dart_api_key_falls_back_to_streamlit_secrets(monkeypatch):
    """환경변수가 비어 있으면 st.secrets를 조회한다 (Streamlit Cloud 경로)."""
    monkeypatch.delenv("DART_API_KEY", raising=False)

    fake_st = types.SimpleNamespace(secrets={"DART_API_KEY": "from-secrets"})
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    assert config.get_dart_api_key() == "from-secrets"


def test_get_dart_api_key_empty_when_nothing_set(monkeypatch):
    monkeypatch.delenv("DART_API_KEY", raising=False)

    # st.secrets 접근 시 파일 없음 예외를 던지는 상황을 흉내낸다.
    class _NoSecrets:
        @property
        def secrets(self):
            raise RuntimeError("no secrets.toml")

    monkeypatch.setitem(sys.modules, "streamlit", _NoSecrets())
    assert config.get_dart_api_key() == ""


@pytest.mark.network
@pytest.mark.skipif(not config.DART_API_KEY, reason="DART_API_KEY가 설정되지 않음")
def test_dart_fetch_disclosures_with_key():
    df = dart.fetch_disclosures("005930", days_back=180)
    assert {"rcept_dt", "report_nm", "flr_nm", "url"} <= set(df.columns)
