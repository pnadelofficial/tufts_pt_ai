import os
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import prompts
import time
import streamlit as st
import re

os.environ["AUTOGEN_USE_DOCKER"] = "False"

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'questions' not in st.session_state:
    st.session_state['questions'] = []

class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        output = message["name"] + ": " + message["content"]
        content = message["content"]
        if '---' in content:
            qs = re.split('---', content)
            for q in qs:
                st.session_state['questions'].append(q.strip())
        st.session_state['messages'].append(output)
        return super()._process_received_message(message, sender, silent)

class TrackableAccepterAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        output = message["name"] + ": " + message["content"]
        content = message["content"]
        pattern = r"Question\s\(Q\):.*Bloom's\sTaxonomy.*|Learning Objective \(LO\):.*Bloom's\sTaxonomy.*"
        q_check = re.search(pattern, content, re.DOTALL)
        if q_check:
            st.session_state['questions'].append(q_check.group(0))
        st.session_state['messages'].append(output)
        return super()._process_received_message(message, sender, silent)
    
class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        output = message["name"] + ": " + message["content"]
        st.session_state['messages'].append(output)
        return super()._process_received_message(message, sender, silent)
    
class TrackableGPTAssistantAgent(GPTAssistantAgent):
    def _process_received_message(self, message, sender, silent):
        output = message["name"] + ": " + message["content"]
        st.session_state['messages'].append(output)
        return super()._process_received_message(message, sender, silent)

class MCQGroupChat:
    def __init__(self, name, agent_dict, init_system_message=prompts.DEFAULT_MCQ_GEN, accepter_model='gpt-4-1106-preview', seed=42, accepter_temp=0, accepter_timeout=120, max_round=10):
        self.name = name
        self.message = init_system_message
        self.agent_dict = agent_dict
        self.accepter_model = accepter_model
        self.seed = seed
        self.accepter_temp = accepter_temp
        self.accepter_timeout = accepter_timeout
        self.max_round = max_round

        self.api_key = st.secrets['openai']["open_ai_key"]

        self.agents = {}
        for name, tup in agent_dict.items():
            agent_id, model, description, system_message = tup
            if agent_id:
                agent = TrackableGPTAssistantAgent(
                    name=name, 
                    llm_config = {
                        "config_list": [
                            {"model":model,"api_key":self.api_key}
                            ], 
                            "assistant_id":agent_id,
                        }
                    )
            else:
                agent = TrackableAssistantAgent(
                    name=name, 
                    llm_config = {
                        "config_list": [
                            {"model":model,"api_key":self.api_key}
                            ]
                        },
                    system_message=system_message,
                    description=description
                    )
            agent.register_function(function_map={})
            self.agents[name] = agent

        self.config_list = [
            {
                "model": self.accepter_model,
                "api_key": self.api_key,
            },
        ]

        self.gpt_config = {
            "cache_seed": self.seed,
            "temperature": self.accepter_temp,
            "config_list": self.config_list,
            "timeout": accepter_timeout,
        }

        self.initializer = TrackableUserProxyAgent(
            name="Init",
            system_message=system_message
        )

        self.accepter = TrackableAccepterAssistantAgent(
            name="Accepter",
            llm_config=self.gpt_config,
            system_message=prompts.ACCEPTER,
            description="""I am **ONLY** allowed to speak **immediately** after `reflect`. I must make sure to output **ONLY** the multiple-choice item and **ONLY** in the appropriate format
    with **ALL** of the appropriate labels. TERMINATE"""
        )
        self.agents["Accepter"] = self.accepter

    def start_chat(self, messages=[], **kwargs):
        self.groupchat = GroupChat(
            agents=self.agents.values(),
            messages=messages,
            max_round=self.max_round,
            **kwargs
        )
        self.manager = GroupChatManager(groupchat=self.groupchat, llm_config=self.gpt_config)
    
    def __call__(self, **kwargs):
        self.start_chat(**kwargs)
        error_counter = 0
        try:
            self.chat = self.initializer.initiate_chat(
                self.manager, message=self.message
            )
        except Exception as e:
            error_counter += 1
            if error_counter < 3:
                print(f"Error: {e}. Retrying...")
                self.chat = self.initializer.initiate_chat(
                    self.manager, message=self.message
                )
            else:
                print(f"Error: {e}. Exiting...")
                return None
        return self.chat
    
class Stateflow(MCQGroupChat):
    def __init__(self, seed=42, **kwargs):
        agent_dict = {
            "QUADL_CAPTE": ("asst_3O5xFFjKdgoIigPlwDrmIlRX", "gpt-4", None, None),
            "s_critic": ("asst_tHvvMNxSo6MbkwJnR3LDTFhV", "gpt-4", None, None),
            "s_critic2": ("asst_6u7I2TCs7adHSS7rOA8I91YA", "gpt-4", None, None),
        }
        super().__init__("Stateflow", agent_dict, seed=seed, **kwargs)

    def state_transition(self, last_speaker, groupchat):
        messages = groupchat.messages

        if last_speaker is self.initializer:
            quadl = self.agents["QUADL_CAPTE"]
            time.sleep(3)
            return quadl
            # State 1-> State 2: Verify
        elif last_speaker is self.agents["QUADL_CAPTE"]:
            # Assuming the last message contains the output path or relevant information
            s_critic = self.agents["s_critic"]
            time.sleep(3)
            return s_critic
        elif last_speaker is self.agents["s_critic"]:
            if messages[-1]["content"].strip().lower() == "true":
                print("Transitioning to accepter")
                return self.accepter
            else:
                print("Transitioning to s_critic2")
                s_critic2 = self.agents["s_critic2"]
                time.sleep(3)
                return s_critic2
        elif last_speaker is self.agents["s_critic2"]:
            if messages[-1]["content"].strip().lower() == "true":
                print("Transitioning to accepter")
                return self.accepter      
            else:
                print("Transitioning back to QUADL_CAPTE")
                quadl = self.agents["QUADL_CAPTE"]
                time.sleep(3)
                return quadl
        elif last_speaker is self.accepter:
            # State 3 -> End
            return None
    
    def __call__(self, **kwargs):
        return super().__call__(speaker_selection_method=self.state_transition, **kwargs)
    
class StateflowNoCritics(MCQGroupChat):
    def __init__(self, seed=42, **kwargs):
        agent_dict = {
            "QUADL_CAPTE": ("asst_3O5xFFjKdgoIigPlwDrmIlRX", "gpt-4", None, None),
        }
        super().__init__("Stateflow", agent_dict, seed=seed, **kwargs)

    def state_transition(self, last_speaker, groupchat):
        messages = groupchat.messages

        if last_speaker is self.initializer:
            quadl = self.agents["QUADL_CAPTE"]
            time.sleep(3)
            return quadl
            # State 1-> State 2: Verify
        elif last_speaker is self.agents["QUADL_CAPTE"]:
            # Assuming the last message contains the output path or relevant information
            accepter = self.agents["Accepter"]
            time.sleep(3)
            return accepter
        elif last_speaker is self.accepter:
            # State 3 -> End
            return None
    
    def __call__(self, **kwargs):
        return super().__call__(speaker_selection_method=self.state_transition, **kwargs)
    
class DebateAndReflect(MCQGroupChat):
    def __init__(self, seed=42, **kwargs):
        agent_dict = {
            "QUADL_CAPTE": ("asst_3O5xFFjKdgoIigPlwDrmIlRX", "gpt-4-1106-preview", None, None),
            "diff_critA":(None, "gpt-3.5-turbo", prompts.DIFF_CRIT, "I am **ONLY** allowed to to speak  **immediately** after `QUADL_CAPTE` or `diff_critB`."),
            "diff_critB":(None, "gpt-3.5-turbo", prompts.DIFF_CRIT, "I am **ONLY** allowed to to speak  **immediately** after `diff_critA`."),
            "diff_critC":(None, "gpt-3.5-turbo", prompts.DIFF_CRIT, "I am **ONLY** allowed to to speak  **immediately** after `diff_critB`."),
            "reflect":(None, "gpt-4-1106-preview", prompts.REFLECTER, "I am **ONLY** allowed to speak **immediately** after `diff_critC`. I will **ONLY** communicate with `accepter`."),
        }
        super().__init__("DebateAndReflect", agent_dict, seed=seed, **kwargs)
        self.agents["user_proxy"] = self.initializer

        self.graph_dict = {}
        self.graph_dict[self.agents["user_proxy"]] = [self.agents["QUADL_CAPTE"]]
        self.graph_dict[self.agents["QUADL_CAPTE"]] = [self.agents["diff_critA"]]
        self.graph_dict[self.agents["diff_critA"]] = [self.agents["diff_critB"]]
        self.graph_dict[self.agents["diff_critB"]] = [self.agents["diff_critA"], self.agents["diff_critC"]]
        self.graph_dict[self.agents["diff_critC"]] = [self.agents["reflect"]]
        self.graph_dict[self.agents["reflect"]] = [self.agents["Accepter"]]
        #peter added
        self.graph_dict[self.agents["Accepter"]] = [self.agents["diff_critA"]]
    
    def start_chat(self, messages=[], **kwargs):
        self.groupchat = GroupChat(
            agents=self.agents.values(),
            messages=messages,
            max_round=self.max_round,
            allowed_or_disallowed_speaker_transitions=self.graph_dict,
            allow_repeat_speaker=None, 
            speaker_transitions_type="allowed",
            **kwargs
        )
        self.manager = GroupChatManager(
            groupchat=self.groupchat, 
            llm_config=self.gpt_config,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
        )