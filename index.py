import faiss
import numpy as np

# 임베딩 로드
embeddings = np.load("output/product_embeddings.npy")

# 차원 확인
dim = embeddings.shape[1]

# FAISS 인덱스 생성
index = faiss.IndexFlatL2(dim)

# 벡터 추가
index.add(embeddings)

# 파일 저장
faiss.write_index(index, "output/excel_search.index")

print("인덱스 생성 완료")