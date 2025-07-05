import streamlit as st
import os

st.title("测试页面")

st.write("当前工作目录:", os.getcwd())

base_path = 'data/generated_poems'
if os.path.exists(base_path):
    st.write(f"{base_path} 下文件夹：", os.listdir(base_path))
else:
    st.error(f"目录 {base_path} 不存在")
