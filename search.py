import os
import json
import torch
import torch.nn.functional as F
import faiss

from datetime import datetime
from transformers import AutoTokenizer, AutoModel


# =========================
# 1. 기본 설정
# =========================

# 빌드할 때 사용한 임베딩 모델과 동일해야 한다.
MODEL_NAME = "BAAI/bge-m3"

# 결과 파일들이 저장된 폴더다.
OUTPUT_DIR = "output"

# FAISS 인덱스 경로다.
INDEX_PATH = os.path.join(OUTPUT_DIR, "excel_search.index")

# 메타데이터 경로다.
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")

# 사용자 선택 로그를 저장할 파일 경로다.
LOG_PATH = os.path.join(OUTPUT_DIR, "search_selection_log.jsonl")

# 상위 몇 개를 보여줄지 정한다.
TOP_K = 5


# =========================
# 2. 임베딩 클래스
# =========================

class BGEEmbedder:
    def __init__(self, model_name):
        # GPU 사용 가능하면 cuda, 아니면 cpu를 사용한다.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")

        # 토크나이저를 불러온다.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 모델을 불러온다.
        self.model = AutoModel.from_pretrained(model_name)

        # 모델을 현재 디바이스로 이동한다.
        self.model.to(self.device)

        # 추론 모드로 전환한다.
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        # 모델 출력에서 토큰 단위 임베딩을 꺼낸다.
        token_embeddings = model_output.last_hidden_state

        # attention mask를 임베딩 텐서 크기에 맞게 확장한다.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 패딩이 아닌 토큰만 반영해서 합계를 구한다.
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # 실제 토큰 개수를 구한다.
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # 평균을 내서 문장 임베딩을 만든다.
        return sum_embeddings / sum_mask

    def encode(self, texts, max_length=512):
        # 문자열 하나만 들어오면 리스트로 감싼다.
        if isinstance(texts, str):
            texts = [texts]

        # 토크나이징한다.
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # 각 텐서를 현재 디바이스로 옮긴다.
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # 추론만 수행한다.
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # 평균 풀링으로 문장 임베딩을 만든다.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])

        # 코사인 유사도 검색을 위해 정규화한다.
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # numpy 대신 faiss search에 넣기 좋은 float32 배열로 바꾼다.
        return sentence_embeddings.cpu().numpy().astype("float32")


# =========================
# 3. 메타데이터 로드
# =========================

def load_metadata(metadata_path):
    # 한 줄씩 읽은 record를 담을 리스트다.
    records = []

    # metadata.jsonl 파일을 연다.
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            # 줄 양쪽 공백 제거
            line = line.strip()

            # 빈 줄이면 건너뛴다.
            if not line:
                continue

            # JSON 문자열을 dict로 바꿔 저장한다.
            records.append(json.loads(line))

    return records


# =========================
# 4. 검색 함수
# =========================

def search(query, embedder, index, metadata, top_k=5):
    # 질문을 임베딩 벡터로 변환한다.
    query_vector = embedder.encode(query)

    # FAISS에서 상위 top_k개를 검색한다.
    scores, indices = index.search(query_vector, top_k)

    # 최종 검색 결과를 담을 리스트다.
    results = []

    # 이번 코드는 질문 1개만 넣으므로 scores[0], indices[0]만 사용한다.
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        # 인덱스 범위가 잘못된 경우는 건너뛴다.
        if idx < 0 or idx >= len(metadata):
            continue

        # 해당 metadata record를 가져온다.
        record = metadata[idx]

        # 결과 리스트에 순위, 점수, 메타데이터를 같이 담는다.
        results.append({
            "rank": rank,
            "score": float(score),
            "metadata_index": int(idx),
            "record": record
        })

    return results


# =========================
# 5. 요약 표시용 값 추출
# =========================

def pick_display_value(raw_data, candidate_keys):
    # raw_data 안에서 candidate_keys 순서대로 찾아
    # 먼저 발견되는 값을 반환한다.
    for key in candidate_keys:
        if key in raw_data:
            value = raw_data.get(key)
            if value is not None and str(value).strip() != "":
                return str(value).strip()

    # 아무것도 못 찾으면 빈 문자열 반환
    return ""


# =========================
# 6. 결과 요약 출력
# =========================

def print_top_results(results):
    # 결과가 없으면 안내 문구 출력
    if not results:
        print("\n검색 결과가 없습니다.")
        return

    print("\n유사한 결과 상위 목록")
    print("=" * 80)

    # 결과를 하나씩 간단히 보여준다.
    for item in results:
        record = item["record"]
        raw_data = record.get("raw_data", {})

        # 보여주기 좋은 주요 필드를 뽑는다.
        product_name = pick_display_value(raw_data, ["상품명", "제품명", "품명", "품목명", "모델명"])
        product_code = pick_display_value(raw_data, ["상품코드", "제품코드", "품목코드", "SKU", "sku"])
        warehouse = pick_display_value(raw_data, ["창고", "물류센터", "보관위치", "저장위치"])
        price = pick_display_value(raw_data, ["가격", "단가", "판매가", "단가(원)", "공급가", "원가"])
        stock = pick_display_value(raw_data, ["재고수량", "수량", "잔여수량", "보유수량", "현재고"])
        category = pick_display_value(raw_data, ["카테고리", "분류", "구분"])

        # 상품명이 없으면 text 일부라도 보여준다.
        if product_name == "":
            product_name = record.get("text", "")[:80]

        print(f"\n[{item['rank']}] 유사도: {item['score']:.4f}")
        print(f"상품명: {product_name}")
        print(f"상품코드: {product_code}")
        print(f"카테고리: {category}")
        print(f"창고: {warehouse}")
        print(f"가격: {price}")
        print(f"재고: {stock}")
        print(f"파일: {record.get('source_file', '')} / 시트: {record.get('sheet_name', '')}")

    print("-" * 80)


# =========================
# 7. 상세 출력
# =========================

def print_selected_detail(selected_item):
    # 선택된 결과의 상세 정보를 보기 좋게 출력한다.
    record = selected_item["record"]
    raw_data = record.get("raw_data", {})

    print("\n선택한 결과 상세")
    print("=" * 80)
    print(f"순위: {selected_item['rank']}")
    print(f"유사도: {selected_item['score']:.4f}")
    print(f"파일명: {record.get('source_file', '')}")
    print(f"시트명: {record.get('sheet_name', '')}")
    print(f"행 번호: {record.get('row_index', '')}")
    print(f"검색용 text: {record.get('text', '')}")
    print("\n원본 데이터:")

    for k, v in raw_data.items():
        print(f"  - {k}: {v}")

    print("-" * 80)


# =========================
# 8. 사용자 선택 받기
# =========================

def ask_user_to_select(results):
    # 결과가 없으면 선택할 수 없으므로 None 반환
    if not results:
        return None

    while True:
        user_input = input(
            "\n이 질문과 가장 가까운 결과 번호를 선택하세요 (1~5, 다시검색=r, 종료=q) >>> "
        ).strip().lower()

        # 종료
        if user_input == "q":
            return "quit"

        # 다시 검색
        if user_input == "r":
            return "retry"

        # 숫자 입력인지 확인
        if not user_input.isdigit():
            print("숫자 번호를 입력하거나 r / q 를 입력하세요.")
            continue

        selected_rank = int(user_input)

        # 범위 검사
        if selected_rank < 1 or selected_rank > len(results):
            print("보여준 번호 범위 안에서 입력하세요.")
            continue

        # rank 기준으로 해당 결과 반환
        for item in results:
            if item["rank"] == selected_rank:
                return item

        print("선택한 번호를 찾지 못했습니다. 다시 입력하세요.")


# =========================
# 9. 선택 로그 저장
# =========================

def save_selection_log(query, results, selected_item, log_path):
    # 로그로 남길 top 결과 요약을 만든다.
    top_results_summary = []

    for item in results:
        record = item["record"]
        top_results_summary.append({
            "rank": item["rank"],
            "score": item["score"],
            "metadata_index": item["metadata_index"],
            "source_file": record.get("source_file"),
            "sheet_name": record.get("sheet_name"),
            "row_index": record.get("row_index"),
            "text": record.get("text")
        })

    # 최종 로그 객체를 만든다.
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "top_results": top_results_summary,
        "selected_rank": selected_item["rank"],
        "selected_score": selected_item["score"],
        "selected_metadata_index": selected_item["metadata_index"],
        "selected_source_file": selected_item["record"].get("source_file"),
        "selected_sheet_name": selected_item["record"].get("sheet_name"),
        "selected_row_index": selected_item["record"].get("row_index"),
        "selected_text": selected_item["record"].get("text")
    }

    # 로그 파일에 JSONL 한 줄로 추가한다.
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


# =========================
# 10. 메인 함수
# =========================

def main():
    # 인덱스 파일 존재 여부 확인
    if not os.path.exists(INDEX_PATH):
        print(f"인덱스 파일이 없습니다: {INDEX_PATH}")
        return

    # 메타데이터 파일 존재 여부 확인
    if not os.path.exists(METADATA_PATH):
        print(f"메타데이터 파일이 없습니다: {METADATA_PATH}")
        return

    # FAISS 인덱스를 읽는다.
    index = faiss.read_index(INDEX_PATH)
    print("FAISS 인덱스 로드 완료")

    # 메타데이터를 읽는다.
    metadata = load_metadata(METADATA_PATH)
    print(f"메타데이터 로드 완료: {len(metadata)}개")

    # 임베딩 모델 로드
    embedder = BGEEmbedder(MODEL_NAME)

    print("\n검색을 시작합니다.")
    print("질문을 입력하면 상위 결과를 보여주고, 번호를 선택할 수 있습니다.")
    print("종료하려면 질문 입력에서 q 를 입력하세요.")

    while True:
        # 질문 입력
        query = input("\n질문 입력 >>> ").strip()

        # 종료
        if query.lower() == "q":
            print("검색을 종료합니다.")
            break

        # 빈 입력 방지
        if query == "":
            print("질문이 비어 있습니다.")
            continue

        # 검색 수행
        results = search(query, embedder, index, metadata, top_k=TOP_K)

        # top 결과 출력
        print_top_results(results)

        # 사용자 선택 받기
        selected = ask_user_to_select(results)

        # 사용자가 q 입력한 경우
        if selected == "quit":
            print("검색을 종료합니다.")
            break

        # 사용자가 다시 검색을 원한 경우
        if selected == "retry":
            continue

        # 정상 선택이면 상세 출력
        print_selected_detail(selected)

        # 로그 저장
        save_selection_log(query, results, selected, LOG_PATH)
        print(f"선택 로그 저장 완료: {LOG_PATH}")


# 직접 실행 시 main() 호출
if __name__ == "__main__":
    main()

    def parse_price_to_int(value):
    # 값이 None이면 비교할 수 없으므로 None 반환
    if value is None:
        return None

    # 문자열로 바꾼다.
    text = str(value).strip()

    # 빈 문자열이면 None
    if text == "":
        return None

    # 쉼표, 원 제거
    text = text.replace(",", "")
    text = text.replace("원", "")
    text = text.strip()

    # 숫자와 점만 남긴다.
    cleaned = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            cleaned += ch

    # 아무 숫자도 없으면 None
    if cleaned == "":
        return None

    # 실수로 바꾼 뒤 int 처리
    try:
        return int(float(cleaned))
    except:
        return None