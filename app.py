import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/ask"


st.set_page_config(
    page_title="Research Agent",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 Research Agent")

st.caption(
    "Local AI Research Assistant · Qwen3 + RAG + Python Data Analysis"
)


# 保存聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []


# 显示历史聊天
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 用户输入
question = st.chat_input(
    "请输入你的问题，例如：分析一下实验数据的变化趋势"
)


if question:

    # 显示用户问题
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # 调用 FastAPI
    with st.chat_message("assistant"):

        with st.spinner("Agent 正在分析..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=300
                )

                result = response.json()


                if "answer" in result:

                    answer = result["answer"]

                elif "error" in result:

                    answer = f"发生错误：{result['error']}"

                else:

                    answer = "API 返回了未知结果。"


            except requests.exceptions.ConnectionError:

                answer = (
                    "无法连接 FastAPI。\n\n"
                    "请确认 FastAPI 已经在 "
                    "`http://127.0.0.1:8000` 运行。"
                )


            except Exception as e:

                answer = f"发生错误：{str(e)}"


        st.markdown(answer)


    # 保存AI回答
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )