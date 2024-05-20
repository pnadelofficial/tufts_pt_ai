import streamlit as st
import chatting
import prompts
import pandas as pd
import re

st.title('Tufts DPT - Multiple Choice Generation')
init_system_message = st.text_area('Starting message:', value=prompts.DEFAULT_MCQ_GEN, height=450)
gc_option = st.selectbox('Group chat option:', ['Debate and Reflect', 'Stateflow', 'Stateflow (no critics)'])

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if gc_option == 'Stateflow':
    chat = chatting.Stateflow(init_system_message=prompts.SYSTEM_MESSAGE) 
    chat.start_chat()
if gc_option == 'Stateflow (no critics)':
    chat = chatting.StateflowNoCritics(init_system_message=prompts.SYSTEM_MESSAGE) 
    chat.start_chat()
if gc_option == 'Debate and Reflect':
    chat = chatting.DebateAndReflect(init_system_message=init_system_message) 
    chat.start_chat()

def clear_state():
    st.session_state['questions'] = []

if st.button('Start Chat', on_click=clear_state):
    with st.spinner('Generating questions...'):
        chat()

    q_dicts = {}
    q_raw = st.session_state['questions'][-1].replace('\n', '')

    questions = re.split(r'(?=Learning)', q_raw)[1:]

    for i, q in enumerate(questions):
        q_dict = {}
        lo_check = re.search(r"(Learning Objective) \(LO\):(.*?)(?=CAPTE)", q)
        if lo_check:
            q_dict[lo_check.group(1)] = lo_check.group(2)
        capte_check = re.search(r"(CAPTE Standard) \(CAPTE\):(.*?)(?=Stem)", q)
        if capte_check:
            q_dict[capte_check.group(1)] = capte_check.group(2)
        stem_check = re.search(r"(Stem):(.*?)(?=Answer)", q)
        if stem_check:
            q_dict[stem_check.group(1)] = stem_check.group(2)    
        answer_check = re.search(r"(Answer) \(A\):(.*?)(?=New Distractors)", q)
        if answer_check:
            q_dict[answer_check.group(1)] = answer_check.group(2)
        distractors_check = re.search(r"(New Distractors):(.*?)(?=Sentence)", q)
        if distractors_check:
            q_dict[distractors_check.group(1)] = distractors_check.group(2)
        sentence_check = re.search(r"(Sentence quote from content).*?:(.*?)(?=Bloom)", q)
        if sentence_check:
            q_dict[sentence_check.group(1)] = sentence_check.group(2)
        bloom_check = re.search(r"(Bloom's Taxonomy Level) \(B\):(.*)", q)
        if bloom_check:
            bloom = bloom_check.group(2)
            bloom = re.sub(r'###.*', '', bloom)
            bloom = re.sub(r'---', '', bloom)
            q_dict[bloom_check.group(1)] = bloom_check.group(2)
        q_dicts[i] = q_dict
    
    for i in range(len(q_dicts)):
        st.write(f'**Question {i+1}**')
        q_dict = q_dicts[i]
        for key, value in q_dict.items():
            if key == 'New Distractors':
                value = re.split(r'[a-z]\)', value)[1:]
                for i, v in enumerate(value):
                    if i == 0:
                        st.write(f'**{"B)"}**: {v}')
                    elif i == 1:
                        st.write(f'**{"C)"}**: {v}')
                    elif i == 2:
                        st.write(f'**{"D)"}**: {v}')
            elif key == 'Answer':
                st.write(f'**{"Correct answer: A)"}**: {value}')
            else: 
                st.write(f'**{key}**: {value}')
        st.divider()