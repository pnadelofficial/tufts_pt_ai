import streamlit as st
import chatting
import prompts
import asyncio

st.title('Tufts DPT - Multiple Choice Generation')
message = st.text_area('Starting message:', value=prompts.DEFAULT_MCQ_GEN)
gc_option = st.selectbox('Group chat option:', ['Stateflow'])

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if gc_option == 'Stateflow':
    chat = chatting.Stateflow(message=message)
    chat.start_chat()
current_len = len(st.session_state['messages'])

if st.button('Start Chat'):
    chat()