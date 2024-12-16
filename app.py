import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA  # 경로 수정
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# 환경 변수 로드 (경로 명시)
load_dotenv(dotenv_path="/home/a09999/projects/rag_project/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit UI 설정
st.title("AI 기반 질의응답 시스템")
st.write("예술가들의 AI 저작권 침해 소송 등 다양한 질문을 입력해 보세요!")

# 벡터스토어 로드
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
st.success("✅ 벡터스토어가 성공적으로 로드되었습니다!")

# RetrievalQA 체인 설정
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="다음 문맥을 바탕으로 질문에 답하세요.\n\n문맥:\n{context}\n\n질문:\n{question}\n\n답변:"
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)

# 사용자 입력
query = st.text_input("질문을 입력하세요:", "")
if st.button("질문하기"):
    if query:
        result = qa_chain({"query": query})

        # Streamlit을 사용해 결과 출력
        st.subheader("🎯 답변:")
        st.write(result["result"])

        # 검색된 문서 확인
        st.subheader("🔍 검색된 문서:")
        for doc in result["source_documents"]:
            st.text(doc.page_content)
    else:
        st.warning("질문을 입력해 주세요!")
