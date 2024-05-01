import streamlit as st
import chatting
import prompts
import pandas as pd
import re

st.title('Tufts DPT - Multiple Choice Generation')
message = st.text_area('Starting message:', value=prompts.DEFAULT_MCQ_GEN)
gc_option = st.selectbox('Group chat option:', ['Stateflow', 'Stateflow (no critics)'])

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if gc_option == 'Stateflow':
    chat = chatting.Stateflow(system_message=prompts.SYSTEM_MESSAGE, message=message) 
    chat.start_chat()
if gc_option == 'Stateflow (no critics)':
    chat = chatting.StateflowNoCritics(system_message=prompts.SYSTEM_MESSAGE, message=message) 
    chat.start_chat()

def clear_state():
    st.session_state['questions'] = []

if st.button('Start Chat', on_click=clear_state):
    with st.spinner('Generating questions...'):
        chat()

    for q in st.session_state['questions']:
        st.header(f'Question {st.session_state["questions"].index(q)+1}')
        _split = [s for s in re.split(r'([A-Z]+:)|([A-Z]+\.)', q)[1:] if s is not None]
        st.markdown('\n'.join(['* '+''.join([s, _split[i+1]]) for i, s in enumerate(_split[:-1]) if i % 2 == 0]))
    st.download_button('Download questions as a CSV', pd.DataFrame(st.session_state['questions']).to_csv(), f'questions.csv', 'text/csv')