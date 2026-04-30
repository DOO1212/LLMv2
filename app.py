import streamlit as st

# 기존 코드 import
from search import (
    load_metadata,
    BGEEmbedder,
    exact_filter_search,
    print_selected_detail,
    save_selection_log,
    LOG_PATH
)

import faiss
import os

# ------------------------
# 초기 로딩 (캐싱)
# ------------------------

@st.cache_resource
def load_system():
    metadata = load_metadata("output/metadata.jsonl")
    embedder = BGEEmbedder("BAAI/bge-m3")
    return metadata, embedder

metadata, embedder = load_system()

# ------------------------
# UI
# ------------------------

st.title("📦 엑셀 검색 챗봇")

query = st.text_input("질문 입력", placeholder="예: 만원 이하 충전기")

# ------------------------
# 검색 버튼
# ------------------------

if st.button("검색"):

    if query.strip() == "":
        st.warning("질문을 입력하세요.")
    else:
        results = exact_filter_search(metadata, query, embedder)

        if not results:
            st.error("검색 결과가 없습니다.")
        else:
            st.session_state["results"] = results
            st.session_state["query"] = query

# ------------------------
# 결과 출력
# ------------------------

if "results" in st.session_state:
    results = st.session_state["results"]

    st.subheader("검색 결과")

    selected_indices = []

    for i, item in enumerate(results):
        record = item["record"]
        raw = record.get("raw_data", {})

        name = raw.get("품목명", "")
        price = raw.get("단가(원)", "")
        warehouse = raw.get("창고", "")

        col1, col2 = st.columns([1, 5])

        with col1:
            if st.checkbox("", key=f"chk_{i}"):
                selected_indices.append(i)

        with col2:
            st.write(f"**{i+1}. {name}**")
            st.write(f"가격: {price} / 창고: {warehouse}")
            st.divider()

    # ------------------------
    # 선택 버튼
    # ------------------------

    if st.button("선택 결과 확인"):

        if not selected_indices:
            st.warning("하나 이상 선택하세요.")
        else:
            st.subheader("선택 결과 상세")

            for i in selected_indices:
                item = results[i]
                record = item["record"]
                raw = record.get("raw_data", {})

                st.write(f"### {raw.get('품목명', '')}")
                for k, v in raw.items():
                    st.write(f"{k}: {v}")

                st.divider()

                # 로그 저장
                save_selection_log(
                    st.session_state["query"],
                    results,
                    item,
                    LOG_PATH
                )

            st.success("로그 저장 완료")