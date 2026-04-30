# os는 폴더와 파일 경로를 다루기 위해 사용한다.
import os

# json은 metadata.jsonl 파일을 저장하기 위해 사용한다.
import json

# pandas는 엑셀 파일을 읽기 위해 사용한다.
import pandas as pd


# 엑셀 파일들이 들어있는 폴더 이름이다.
DATA_DIR = "data"

# 결과 파일을 저장할 폴더 이름이다.
OUTPUT_DIR = "output"

# metadata.jsonl 파일 저장 경로다.
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.jsonl")


# 한 행을 검색용 텍스트로 바꾸는 함수다.
def row_to_text(row):
    # "컬럼명: 값" 형태의 문자열 조각을 담을 리스트다.
    parts = []

    # 현재 행의 모든 컬럼명을 하나씩 확인한다.
    for col in row.index:
        # 현재 컬럼의 값을 가져온다.
        value = row[col]

        # 값이 비어 있으면 건너뛴다.
        if pd.isna(value):
            continue

        # 컬럼명과 값을 "컬럼명: 값" 형태로 만든다.
        part = f"{col}: {value}"

        # 만든 문자열 조각을 리스트에 추가한다.
        parts.append(part)

    # 여러 조각을 하나의 긴 문자열로 합쳐 반환한다.
    return " / ".join(parts)


# 엑셀 파일들을 읽어서 metadata를 만드는 함수다.
def build_metadata():
    # 최종 metadata 레코드들을 담을 리스트다.
    records = []

    # output 폴더가 없으면 새로 만든다.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # data 폴더가 없으면 에러 메시지를 출력하고 종료한다.
    if not os.path.exists(DATA_DIR):
        print(f"{DATA_DIR} 폴더가 없습니다.")
        return

    # data 폴더 안의 파일명을 하나씩 확인한다.
    for file_name in os.listdir(DATA_DIR):
        # 임시 파일(~$)은 건너뛴다.
        if file_name.startswith("~$"):
            continue

        # 엑셀 파일만 처리한다.
        if not file_name.endswith(".xlsx"):
            continue
        # xlsx 파일만 처리한다.
        if not file_name.endswith(".xlsx"):
            continue

        # 엑셀 파일 전체 경로를 만든다.
        file_path = os.path.join(DATA_DIR, file_name)

        # 현재 읽는 파일명을 출력한다.
        print(f"읽는 중: {file_path}")

        # 엑셀 파일 객체를 만든다.
        excel_file = pd.ExcelFile(file_path)

        # 엑셀 파일 안의 모든 시트를 하나씩 처리한다.
        for sheet_name in excel_file.sheet_names:
            # 현재 시트를 데이터프레임으로 읽는다.
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 시트가 비어 있으면 건너뛴다.
            if df.empty:
                continue

            # 데이터프레임의 각 행을 하나씩 처리한다.
            for row_idx, row in df.iterrows():
                # 원본 행 데이터를 dict로 변환한다.
                raw_data = row.to_dict()

                # 검색용 텍스트를 만든다.
                text = row_to_text(row)

                # metadata 한 건을 만든다.
                record = {
                    "source_file": file_name,
                    "sheet_name": sheet_name,
                    "row_index": int(row_idx) + 2,
                    "text": text,
                    "raw_data": raw_data
                }

                # metadata 리스트에 추가한다.
                records.append(record)

    # metadata.jsonl 파일을 저장한다.
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        # metadata를 한 줄씩 JSON으로 저장한다.
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # 완료 메시지를 출력한다.
    print(f"metadata 생성 완료: {len(records)}개")
    print(f"저장 위치: {METADATA_PATH}")


# 이 파일을 직접 실행하면 build_metadata()를 실행한다.
if __name__ == "__main__":
    build_metadata()