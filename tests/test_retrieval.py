from app.rag.chunker import chunk_repository
from app.rag.local_index import LocalIndex, tokenize


def test_tokenize_splits_identifiers_but_keeps_whole_name():
    toks = tokenize("merge_environment_settings getHTTPResponse")
    assert "merge_environment_settings" in toks
    assert {"merge", "environment", "settings", "get", "http", "response"} <= set(toks)


def test_issue_text_retrieves_the_buggy_function_first(wealth_repo):
    idx = LocalIndex(chunk_repository(wealth_repo))
    hits = idx.search("SIP future value is wildly overstated; required_monthly_sip also wrong", top_k=5)
    assert hits[0].file_path == "wealth/sip.py"
    assert hits[0].name in {"future_value", "required_monthly_sip"}


def test_symbol_mention_beats_generic_words(wealth_repo):
    idx = LocalIndex(chunk_repository(wealth_repo))
    hits = idx.search("rebalance_orders sells the wrong amount", top_k=3)
    assert hits[0].name == "rebalance_orders"


def test_hyde_adds_a_ranking(wealth_repo):
    idx = LocalIndex(chunk_repository(wealth_repo))
    query = "buy and sell amounts are off after the mix changes"
    plain = idx.search(query, top_k=3)
    hyde = idx.search(query, top_k=3,
                      hyde_code="def rebalance_orders(holdings, target):\n    current = allocation(holdings)")
    assert hyde[0].name == "rebalance_orders"
    # HyDE code mentions allocation(); prose alone never surfaces it.
    assert "allocation" not in [h.name for h in plain]
    assert "allocation" in [h.name for h in hyde]


def test_chunk_paths_use_forward_slashes(wealth_repo):
    assert all("\\" not in c.file_path for c in chunk_repository(wealth_repo))
