import streamlit as st
import prompts
import utils
import json

st.title('Tufts DPT - Multiple Choice Generation')

uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=False)
if not uploaded_files:
    st.warning('Please upload a PDF file.')
else:
    st.success('PDF file uploaded successfully!')
    difficulty_level = st.selectbox('Select the difficulty level:', ['Very easy', 'Easy', 'Medium', 'Hard', 'Very hard'],index=None)
    if st.button("Generate Question") and (difficulty_level):
        text, init_system_message = utils.file_uploader(uploaded_files, difficulty_level=difficulty_level)
        client = utils.get_client()

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        if 'questions' not in st.session_state:
            st.session_state['questions'] = []

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        if "chat_model" not in st.session_state:
            st.session_state["chat_model"] = "gpt-4o"

        q_dict = utils.produce_question(init_system_message)
        @st.experimental_fragment
        def chat_fragment(q_dict):
            st.session_state['chat_messages'].append(
                {'role':'system', 'content':prompts.CHAT_SYSTEM_PROMPT.format(mcq=json.dumps(q_dict), content=text)}
            )
            chat_container = st.container(height=700)
            beginning_response = client.chat.completions.create(model=st.session_state["chat_model"], messages=st.session_state['chat_messages'])
            beginning_msg = beginning_response.choices[0].message.content
            chat_container.chat_message("assistant").write(beginning_msg) 
            if prompt := st.chat_input():
                st.session_state['chat_messages'].append({"role": "user", "content": prompt})
                chat_container.chat_message("user").write(prompt)
                
                with chat_container.chat_message("assistant"):
                    stream = client.chat.completions.create(model=st.session_state["chat_model"], messages=st.session_state['chat_messages'], stream=True)
                    response = st.write_stream(stream)
                
                st.session_state['chat_messages'].append({"role": "assistant", "content": response})
        chat_fragment(q_dict)
        st.button("Clear chat", on_click=utils.clear_state)
    else:
        st.warning('Please select a difficulty level.')