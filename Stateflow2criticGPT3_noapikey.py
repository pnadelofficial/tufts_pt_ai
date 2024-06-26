import os
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# Assuming the environment setup and configurations (gpt4_config) are defined elsewhere

load_dotenv()

os.environ["AUTOGEN_USE_DOCKER"] = "False"
api_key = os.environ["OPENAI_API_KEY"]

config_list_gpt4 = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": api_key,
        # ... other configuration parameters ...
    },
    # ... potentially other configuration dictionaries ...
]
config_list = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": api_key,
        # ... other configuration parameters ...
    },
    # ... potentially other configuration dictionaries ...
]
# ----------------- #

llm_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "config_list": config_list,
    "timeout": 120,
}

gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}


initializer = autogen.UserProxyAgent(
    name="Init",
)

QUADL_CAPTE = GPTAssistantAgent(name="QUADL", llm_config = {"config_list": [{"model":"gpt-4","api_key":api_key}], "assistant_id":"asst_3O5xFFjKdgoIigPlwDrmIlRX"})
QUADL_CAPTE.register_function(function_map={})

s_critic = GPTAssistantAgent(name="s_crit", llm_config = {"config_list": [{"model":"gpt-4","api_key":api_key}], "assistant_id":"asst_tHvvMNxSo6MbkwJnR3LDTFhV"})
s_critic.register_function(function_map={})

s_critic2 = GPTAssistantAgent(name="s_crit", llm_config = {"config_list": [{"model":"gpt-4","api_key":api_key}], "assistant_id":"asst_6u7I2TCs7adHSS7rOA8I91YA"})
s_critic2.register_function(function_map={})

accepter = autogen.AssistantAgent(
    name="Accepter",
    llm_config=gpt4_config,
    system_message="""You are the Accepter agent. Your task is to accept the sentence (S) generated by QUADL_CAPTE as valid and output the 5 multiple-choice items in the appropriate format, including labels: Question(Q), Learning Objective(LO), CAPTE Statndard(CAPTE), Answer(A), Sentence quote from content(S), Bloom(B)."""
)

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is initializer:
        return QUADL_CAPTE
        # State 1-> State 2: Verify
    elif last_speaker is QUADL_CAPTE:
        # Assuming the last message contains the output path or relevant information
         return s_critic
    elif last_speaker is s_critic:
        if messages[-1]["content"].strip().lower() == "true":
            print("Transitioning to accepter")
            return accepter
        else:
            print("Transitioning to s_critic2")
            return s_critic2
    elif last_speaker is s_critic2:
        if messages[-1]["content"].strip().lower() == "true":
            print("Transitioning to accepter")
            return accepter        
        else:
            print("Transitioning back to QUADL_CAPTE")
            return QUADL_CAPTE
    elif last_speaker is accepter:
        # State 3 -> End
        return None

# Assuming 'groupchat' and 'manager' initialization happens correctly elsewhere

groupchat = autogen.GroupChat(
    agents=[initializer, QUADL_CAPTE, s_critic, s_critic2, accepter],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

chat = initializer.initiate_chat(
    manager, message="Create 5 multiple-choice items and output them in the appropriate format, including labels: Learning Objective(LO), CAPTE Statndard(CAPTE), Answer(A), Sentence quote from content(S), Bloom(B)"
)
print(chat)