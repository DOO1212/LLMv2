# jsonl 파일을 읽고 쓰기 위해 json을 가져온다.
import json

# 로그 파일 경로다.
LOG_PATH = "output/search_selection_log.jsonl"

# 학습 데이터 저장 경로다.
TRAIN_PATH = "output/query_parser_train.jsonl"


# 질문에서 단순하게 가격 조건을 추출하는 함수다.
def extract_simple_price_condition(query):
    # 이하 조건이면 <= 로 본다.
    if "이하" in query:
        op = "<="
    # 이상 조건이면 >= 로 본다.
    elif "이상" in query:
        op = ">="
    # 미만 조건이면 < 로 본다.
    elif "미만" in query:
        op = "<"
    # 초과 조건이면 > 로 본다.
    elif "초과" in query:
        op = ">"
    # 조건 단어가 없으면 None을 반환한다.
    else:
        return None

    # 질문을 공백 기준으로 나눈다.
    tokens = query.split()

    # 숫자를 찾는다.
    for token in tokens:
        # 토큰에서 숫자만 남긴다.
        number_text = "".join(ch for ch in token if ch.isdigit())

        # 숫자가 있으면 조건으로 반환한다.
        if number_text:
            return {
                "column": "단가(원)",
                "operator": op,
                "value": int(number_text)
            }

    # 숫자를 못 찾으면 None을 반환한다.
    return None


# 질문에서 상품 키워드를 임시로 추출하는 함수다.
def extract_product_keyword(query):
    # 비교/가격 관련 단어는 키워드에서 제외한다.
    stopwords = ["단가", "가격", "원", "이하", "이상", "미만", "초과"]

    # 질문을 공백 기준으로 나눈다.
    tokens = query.split()

    # 후보 키워드를 저장한다.
    candidates = []

    # 토큰을 하나씩 확인한다.
    for token in tokens:
        # 숫자가 들어간 토큰은 제외한다.
        if any(ch.isdigit() for ch in token):
            continue

        # 불용어가 포함된 토큰은 제외한다.
        if any(stopword in token for stopword in stopwords):
            continue

        # 남은 토큰을 후보로 추가한다.
        candidates.append(token)

    # 후보가 없으면 None을 반환한다.
    if not candidates:
        return None

    # 마지막 후보를 상품 키워드로 본다.
    return candidates[-1]


# 로그를 학습 데이터로 바꾸는 함수다.
def build_training_data():
    # 생성된 학습 데이터를 담을 리스트다.
    rows = []

    # 로그 파일을 연다.
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        # 로그를 한 줄씩 읽는다.
        for line in f:
            # 앞뒤 공백을 제거한다.
            line = line.strip()

            # 빈 줄이면 건너뛴다.
            if not line:
                continue

            # JSON 문자열을 파이썬 dict로 바꾼다.
            log = json.loads(line)

            # 사용자 질문을 가져온다.
            query = log.get("query", "")

            # 필터 목록을 만든다.
            filters = []

            # 가격 조건을 추출한다.
            price_filter = extract_simple_price_condition(query)

            # 가격 조건이 있으면 filters에 추가한다.
            if price_filter:
                filters.append(price_filter)

            # 상품 키워드를 추출한다.
            product_keyword = extract_product_keyword(query)

            # 상품 키워드가 있으면 filters에 추가한다.
            if product_keyword:
                filters.append({
                    "column": "품목명",
                    "operator": "contains",
                    "value": product_keyword
                })

            # 정답 JSON을 만든다.
            output = {
                "filters": filters
            }

            # 학습 데이터 한 줄을 만든다.
            row = {
                "input": query,
                "output": output
            }

            # 리스트에 추가한다.
            rows.append(row)

    # 학습 데이터 파일을 저장한다.
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        # 한 줄씩 JSONL로 저장한다.
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 생성 개수를 출력한다.
    print(f"학습 데이터 생성 완료: {len(rows)}개")
    print(f"저장 위치: {TRAIN_PATH}")


# 이 파일을 직접 실행하면 build_training_data()를 실행한다.
if __name__ == "__main__":
    build_training_data()