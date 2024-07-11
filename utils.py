import streamlit as st
import autogen
import prompts
import os
import re
import pypdf
from openai import OpenAI

os.environ["AUTOGEN_USE_DOCKER"] = "False"

def clear_state():
    st.session_state['questions'] = []
    st.session_state['messages'] = []
    st.session_state['chat_messages'] = []
    st.cache_resource.clear()

def create_config(model):
    return [
        {
            'model': model,
            'api_key': st.secrets['openai']["open_ai_key"]
        }
    ]

@st.cache_resource
def file_uploader(uploaded_files):
    if uploaded_files:
        pdf = pypdf.PdfReader(uploaded_files)
        text = ' '.join([page.extract_text() for page in pdf.pages])
        init_system_message = prompts.INIT_MESSAGE.format(num_questions=1, text=text)
    return text, init_system_message

@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets['openai']["open_ai_key"])

class TrackableAssistantAgent(autogen.AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        output = message["name"] + ": " + message["content"]
        content = message["content"]
        if '---' in content:
            qs = re.split('---', content)
            for q in qs:
                st.session_state['questions'].append(q.strip())
        st.session_state['messages'].append(output)
        return super()._process_received_message(message, sender, silent)

class TrackableAccepterAssistantAgent(autogen.AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        output = message["name"] + ": " + message["content"]
        content = message["content"]
        pattern = r"Question\s\(Q\):.*Bloom's\sTaxonomy.*|Learning Objective \(LO\):.*Bloom's\sTaxonomy.*"
        q_check = re.search(pattern, content, re.DOTALL)
        if q_check:
            st.session_state['questions'].append(q_check.group(0))
        st.session_state['messages'].append(output)
        return super()._process_received_message(message, sender, silent)
    
class TrackableUserProxyAgent(autogen.UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        output = message["name"] + ": " + message["content"]
        st.session_state['messages'].append(output)
        return super()._process_received_message(message, sender, silent)

## agents
@st.cache_resource
def get_agents():
    user_proxy = TrackableUserProxyAgent(
        name="Admin",
        system_message="A human admin",
        code_execution_config=False,
        human_input_mode="NEVER",
        llm_config=False,
        description="Never select me as a speaker.",
    )

    quadl_capte = TrackableAssistantAgent(
        name='QUADL', 
        system_message= prompts.DEFAULT_MCQ_GEN,
        llm_config={"config_list": create_config('gpt-4o'), "cache_seed": None},
    )
    quadl_capte.register_function(function_map={})

    diff_critA = TrackableAssistantAgent(
        name='diff_critA',
        system_message= prompts.DISTRACTOR_A,
        llm_config={"config_list": create_config('gpt-3.5-turbo'), "cache_seed": None},
    )

    diff_critB = TrackableAssistantAgent(
        name='diff_critB',
        system_message= prompts.DISTRACTOR_B,
        llm_config={"config_list": create_config('gpt-3.5-turbo'), "cache_seed": None},
    )

    diff_critC = TrackableAssistantAgent(
        name='diff_critC',
        system_message= prompts.DISTRACTOR_C,
        llm_config={"config_list": create_config('gpt-3.5-turbo'), "cache_seed": None},
    )

    flaws = TrackableAssistantAgent(
        name='flaws',
        system_message= prompts.FLAWS,
        llm_config={"config_list": create_config('gpt-4o'), "cache_seed": None},
    )

    accepter = TrackableAccepterAssistantAgent(
        name='accepter',
        system_message= prompts.ACCEPTER,
        llm_config={"config_list": create_config('gpt-4o'), "cache_seed": None},
    )

    reflect = TrackableAssistantAgent(
        name='reflect',
        system_message= prompts.REFLECT,
        llm_config={"config_list": create_config('gpt-4o'), "cache_seed": None},
    )
    return user_proxy, quadl_capte, diff_critA, diff_critB, diff_critC, flaws, accepter, reflect

@st.cache_resource
def init_chat():
    user_proxy, quadl_capte, diff_critA, diff_critB, diff_critC, flaws, accepter, reflect = get_agents()
    
    graph_dict = {}
    graph_dict[user_proxy] = [quadl_capte]
    graph_dict[quadl_capte] = [diff_critA]
    graph_dict[diff_critA] = [diff_critB]
    graph_dict[diff_critB] = [diff_critC]
    graph_dict[diff_critC] = [flaws]
    graph_dict[flaws] = [reflect]
    graph_dict[reflect] = [accepter]

    groupchat = autogen.GroupChat(agents=[user_proxy, quadl_capte, diff_critA, diff_critB, diff_critC, flaws, accepter, reflect], messages=[], max_round=25, allowed_or_disallowed_speaker_transitions=graph_dict, allow_repeat_speaker=None, speaker_transitions_type="allowed")
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": create_config('gpt-4o')},
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
    )
    return user_proxy, manager

@st.cache_resource
def produce_question(init_system_message):
    with st.spinner('Generating questions...'):
        user_proxy, manager = init_chat()
        user_proxy.initiate_chat(
            manager,
            message=init_system_message,
        )
    
    q_raw = st.session_state['questions'][-1].replace('\n', '')

    questions = re.split(r'(?=Learning)', q_raw)[1:]

    q_dict = {}
    for _, q in enumerate(questions):
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
            q_dict[distractors_check.group(1)] = re.split(r'[bcd]\)', q_dict[distractors_check.group(1)])[1:]
        sentence_check = re.search(r"(Sentence quote from content).*?:(.*?)(?=Bloom)", q)
        if sentence_check:
            q_dict[sentence_check.group(1)] = sentence_check.group(2)
        bloom_check = re.search(r"(Bloom's Taxonomy Level) \(B\):(.*)", q)
        if bloom_check:
            bloom = bloom_check.group(2)
            bloom = re.sub(r'###.*', '', bloom)
            bloom = re.sub(r'---', '', bloom)
            q_dict[bloom_check.group(1)] = bloom_check.group(2)

    return q_dict
