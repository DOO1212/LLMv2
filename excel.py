# 여러 엑셀 파일을 읽어서
# 컬럼 자동 매핑 -> text 생성 -> bge-m3 임베딩 -> FAISS 인덱스 저장까지 하는 전체 스크립트다.

import os
import re
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

print("torch 버전:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda 사용 가능:", torch.cuda.is_available())
print("gpu 개수:", torch.cuda.device_count())
print("gpu 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")


# =========================
# 1. 기본 설정
# =========================

# 엑셀 파일들이 들어있는 폴더 경로다.
DATA_DIR = "DATA"

# 결과물을 저장할 폴더다.
OUTPUT_DIR = "output"

# 사용할 임베딩 모델 이름이다.
MODEL_NAME = "BAAI/bge-m3"

# 한 번에 몇 개 문장을 임베딩할지 정하는 배치 크기다.
# GPU 메모리가 작으면 이 값을 줄이면 된다.
BATCH_SIZE = 16

# 검색 시 text로 만들 때 우선적으로 보고 싶은 논리 컬럼 순서다.
# 실제 엑셀 컬럼명이 아니라, 우리가 통일해서 붙일 표준 이름이다.
TARGET_FIELDS = ["product_name", "price", "description", "stock", "category", "brand"]


# output 폴더가 없으면 생성한다.
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. 컬럼 자동 매핑용 사전
# =========================

# 서로 다른 엑셀 파일에서 제각각 쓰인 컬럼명을
# 하나의 표준 컬럼명으로 묶기 위한 동의어 사전이다.
COLUMN_SYNONYMS = {
    "product_name": [
        "상품명", "제품명", "품명", "상품", "제품", "모델명", "모델", "name", "product", "item"
    ],
    "price": [
        "가격", "금액", "단가", "판매가", "원가", "비용", "price", "cost", "amount"
    ],
    "description": [
        "설명", "상세설명", "상품설명", "비고", "메모", "description", "desc", "note"
    ],
    "stock": [
        "재고", "재고수량", "수량", "잔여수량", "보유수량", "stock", "qty", "quantity"
    ],
    "category": [
        "카테고리", "분류", "구분", "category", "type"
    ],
    "brand": [
        "브랜드", "제조사", "회사", "brand", "maker", "manufacturer"
    ]
}


# =========================
# 3. 컬럼명 정리 함수
# =========================

def normalize_text(text):
    # None 같은 값이 들어와도 문자열로 안전하게 바꾼다.
    text = str(text)

    # 앞뒤 공백을 제거한다.
    text = text.strip()

    # 모두 소문자로 바꿔서 영문 비교를 쉽게 한다.
    text = text.lower()

    # 공백, 언더바, 하이픈을 제거해서 비교를 단순화한다.
    # 예: "상품 명", "상품_명", "상품-명" -> "상품명"
    text = re.sub(r"[\s_\-]+", "", text)

    return text


# =========================
# 4. 실제 컬럼명을 표준 컬럼명으로 추정하는 함수
# =========================

def map_column_name(col_name):
    # 원본 컬럼명을 정규화한다.
    norm_col = normalize_text(col_name)

    # 미리 정의한 표준 컬럼 후보들을 하나씩 본다.
    for target_field, synonym_list in COLUMN_SYNONYMS.items():
        # 각 표준 컬럼에 대응되는 동의어를 하나씩 확인한다.
        for syn in synonym_list:
            # 동의어도 같은 방식으로 정규화한다.
            norm_syn = normalize_text(syn)

            # 컬럼명과 동의어가 완전히 같거나,
            # 컬럼명 안에 동의어가 포함되어 있으면 같은 의미로 본다.
            # 예: "상품명(국문)" 안에 "상품명" 포함
            if norm_col == norm_syn or norm_syn in norm_col:
                return target_field

    # 어떤 표준 컬럼에도 매핑되지 않으면 None을 반환한다.
    return None


# =========================
# 5. row를 text로 바꾸는 함수
# =========================

def row_to_text(row, mapped_columns):
    # 최종적으로 붙일 "컬럼명: 값" 조각들을 저장할 리스트다.
    parts = []

    # 먼저 TARGET_FIELDS 순서대로 중요한 컬럼부터 넣는다.
    for field in TARGET_FIELDS:
        # mapped_columns는 {원본컬럼명: 표준컬럼명} 구조다.
        for original_col, mapped_field in mapped_columns.items():
            # 현재 원본컬럼이 우리가 찾는 표준 컬럼과 일치하면 처리한다.
            if mapped_field == field:
                val = row.get(original_col, None)

                # 비어 있는 값은 건너뛴다.
                if pd.isna(val):
                    continue

                val = str(val).strip()

                # 빈 문자열도 건너뛴다.
                if val == "":
                    continue

                # 예: "product_name: 사과"
                parts.append(f"{field}: {val}")

    # 그 다음, 자동 매핑되지 않은 나머지 컬럼도 뒤에 붙인다.
    # 이렇게 하면 정보 손실이 줄어든다.
    for original_col in row.index:
        if original_col in mapped_columns:
            continue

        val = row.get(original_col, None)

        if pd.isna(val):
            continue

        val = str(val).strip()

        if val == "":
            continue

        parts.append(f"{original_col}: {val}")

    # 최종적으로 " / " 로 이어 붙여 하나의 문장처럼 만든다.
    return " / ".join(parts)


# =========================
# 6. 여러 엑셀 파일 읽기
# =========================

def load_all_excel_rows(data_dir):
    # 모든 row 정보를 담을 리스트다.
    all_records = []

    # DATA 폴더 안의 xlsx, xls 파일을 모두 찾는다.
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    excel_files += glob.glob(os.path.join(data_dir, "*.xls"))

    print(f"발견한 엑셀 파일 수: {len(excel_files)}")

    # 파일 하나씩 처리한다.
    for file_path in excel_files:
        print(f"\n처리 중 파일: {file_path}")

        try:
            # 엑셀의 모든 시트를 읽기 위해 ExcelFile 객체를 만든다.
            xls = pd.ExcelFile(file_path)

            # 시트 이름을 하나씩 순회한다.
            for sheet_name in xls.sheet_names:
                print(f"  - 시트 처리 중: {sheet_name}")

                try:
                    # 현재 시트를 DataFrame으로 읽는다.
                    df = pd.read_excel(file_path, sheet_name=sheet_name)

                    # 완전히 빈 시트면 건너뛴다.
                    if df.empty:
                        print("    -> 빈 시트라서 건너뜀")
                        continue

                    # 컬럼 자동 매핑 결과를 저장할 딕셔너리다.
                    # 예: {"제품명": "product_name", "판매가": "price"}
                    mapped_columns = {}

                    # 현재 시트의 컬럼들을 하나씩 보면서 표준 컬럼으로 매핑한다.
                    for col in df.columns:
                        mapped = map_column_name(col)
                        if mapped is not None:
                            mapped_columns[col] = mapped

                    print(f"    -> 자동 매핑 결과: {mapped_columns}")

                    # DataFrame의 각 row를 하나씩 처리한다.
                    for row_idx, row in df.iterrows():
                        # 현재 row를 검색용 text로 변환한다.
                        text = row_to_text(row, mapped_columns)

                        # text가 비어 있으면 저장하지 않는다.
                        if text.strip() == "":
                            continue

                        # 메타데이터를 함께 저장한다.
                        record = {
                            "source_file": os.path.basename(file_path),
                            "sheet_name": sheet_name,
                            "row_index": int(row_idx),
                            "text": text,
                            "mapped_columns": mapped_columns
                        }

                        # 원본 row 데이터도 dict로 저장한다.
                        # 나중에 검색 결과 보여줄 때 유용하다.
                        raw_data = {}
                        for col in df.columns:
                            val = row[col]

                            # NaN은 JSON 저장이 불편하므로 None으로 바꾼다.
                            if pd.isna(val):
                                raw_data[str(col)] = None
                            else:
                                raw_data[str(col)] = str(val)

                        record["raw_data"] = raw_data

                        all_records.append(record)

                except Exception as e:
                    print(f"    -> 시트 처리 실패: {sheet_name}, 오류: {e}")

        except Exception as e:
            print(f"  -> 파일 처리 실패: {file_path}, 오류: {e}")

    return all_records


# =========================
# 7. BGE-M3 임베딩 클래스
# =========================

class BGEEmbedder:
    def __init__(self, model_name):
        # GPU가 가능하면 cuda, 아니면 cpu를 쓴다.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")

        # 토크나이저를 로드한다.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 모델을 로드한다.
        self.model = AutoModel.from_pretrained(model_name)

        # 모델을 GPU 또는 CPU로 이동시킨다.
        self.model.to(self.device)

        # 추론 모드로 사용하므로 eval 상태로 바꾼다.
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        # 모델의 마지막 hidden states를 가져온다.
        token_embeddings = model_output.last_hidden_state

        # attention mask를 float 형태로 바꾸고, 임베딩 크기에 맞게 차원을 늘린다.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 패딩이 아닌 토큰만 평균에 반영되도록 mask를 곱한다.
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # 각 문장에서 실제 토큰 수를 구한다.
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # 최종적으로 토큰 평균값을 문장 임베딩으로 사용한다.
        return sum_embeddings / sum_mask

    def encode(self, texts, batch_size=16, max_length=512):
        # 모든 배치 결과를 모을 리스트다.
        all_embeddings = []

        # 배치 단위로 text를 나눈다.
        for start_idx in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성 중"):
            batch_texts = texts[start_idx:start_idx + batch_size]

            # 토크나이징한다.
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # 토큰 텐서를 모델 디바이스로 옮긴다.
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # gradient 계산 없이 추론한다.
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # 평균 풀링으로 문장 임베딩을 만든다.
            sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])

            # 코사인 유사도 검색을 위해 L2 정규화한다.
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            # CPU numpy 배열로 바꿔 저장한다.
            all_embeddings.append(sentence_embeddings.cpu().numpy())

        # 배치별 결과를 하나로 합친다.
        all_embeddings = np.vstack(all_embeddings).astype("float32")

        return all_embeddings


# =========================
# 8. 메인 실행 함수
# =========================

def main():
    # 1) 모든 엑셀 row를 읽는다.
    records = load_all_excel_rows(DATA_DIR)

    print(f"\n최종 수집된 row 수: {len(records)}")

    if len(records) == 0:
        print("처리할 데이터가 없습니다.")
        return

    # 2) text만 따로 뽑는다.
    texts = [r["text"] for r in records]

    # 3) 임베딩 모델을 로드한다.
    embedder = BGEEmbedder(MODEL_NAME)

    # 4) text 전체를 임베딩한다.
    embeddings = embedder.encode(texts, batch_size=BATCH_SIZE)

    print(f"임베딩 shape: {embeddings.shape}")

    # 5) FAISS 인덱스를 만든다.
    # 정규화된 벡터 + Inner Product 조합이면 코사인 유사도처럼 사용할 수 있다.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # 6) 벡터를 인덱스에 추가한다.
    index.add(embeddings)

    # 7) FAISS 인덱스를 파일로 저장한다.
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "excel_search.index"))

    # 8) 메타데이터를 JSONL로 저장한다.
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 9) 임베딩 배열도 따로 저장한다.
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)

    print("\n빌드 완료")
    print(f"- 인덱스 저장: {os.path.join(OUTPUT_DIR, 'excel_search.index')}")
    print(f"- 메타데이터 저장: {metadata_path}")
    print(f"- 임베딩 저장: {os.path.join(OUTPUT_DIR, 'embeddings.npy')}")


# 이 파일을 직접 실행했을 때 main()을 실행한다.
if __name__ == "__main__":
    main()