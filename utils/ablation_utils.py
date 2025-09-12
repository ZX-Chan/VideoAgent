import yaml
import json
from utils.wei_utils import account_token
from jinja2 import Environment, StrictUndefined
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from utils.src.utils import get_json_from_response

def no_tree_get_layout(poster_width, poster_height, panels, figures, agent_config):
    total_input_token, total_output_token = 0, 0
    agent_name = 'ablation_all_info_layout'
    with open(f"utils/prompt_templates/{agent_name}.yaml", "r") as f:
        planner_config = yaml.safe_load(f)

    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(planner_config["template"])
    planner_jinja_args = {
        'ppt_width': poster_width,
        'ppt_height': poster_height,
        'panels': json.dumps(panels, indent=4),
        'figures': json.dumps(figures, indent=4),
    }

    planner_model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
    )

    planner_agent = ChatAgent(
        system_message=planner_config['system_prompt'],
        model=planner_model,
        message_window_size=None,
    )

    planner_prompt = template.render(**planner_jinja_args)

    num_trials = 0
    
    while True:
        num_trials += 1
        print(f"Trial {num_trials}: Generating layout...")
        planner_agent.reset()
        response = planner_agent.step(planner_prompt)
        input_token, output_token = account_token(response)
        total_input_token += input_token
        total_output_token += output_token
        print(response.msgs[0].content)
        arrangements = get_json_from_response(response.msgs[0].content)

        if len(arrangements) == 0:
            print('Error: Empty response, retrying...')
            continue

        if not 'slide_layouts' in arrangements or\
        not 'figure_layouts' in arrangements or\
        not 'text_layouts' in arrangements:
            print('Error: Invalid response, retrying...')
            continue


        break
    return arrangements['slide_layouts'], arrangements['figure_layouts'], arrangements['text_layouts'], input_token, output_token