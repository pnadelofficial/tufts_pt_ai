import streamlit as st
import chatting
import prompts
import pandas as pd

st.title('Tufts DPT - Multiple Choice Generation')
message = st.text_area('Starting message:', value=prompts.DEFAULT_MCQ_GEN)
gc_option = st.selectbox('Group chat option:', ['Stateflow'])

if gc_option == 'Stateflow':
    chat = chatting.Stateflow(system_message=prompts.SYSTEM_MESSAGE, message=message) 
    chat.start_chat()

def clear_state():
    st.session_state['questions'] = []

if st.button('Start Chat', on_click=clear_state):
    with st.spinner('Generating questions...'):
        chat()

    for q in st.session_state['questions']:
        st.header(f'Question {st.session_state["questions"].index(q)+1}')
        st.write(q)
    st.download_button('Download questions as a CSV', pd.DataFrame(st.session_state['questions']).to_csv(), f'questions.csv', 'text/csv')