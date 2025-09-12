import os
import json
import numpy as np
import re
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from utils.generate_prompts import ciritique_generation, ciritique_last_generation, ciritique_last_generation_math

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='data/gpqa_diamond_test.json')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--momentum_rate', type=float, default=0.9)
parser.add_argument('--scaling_rate', type=float, default=0.9)
parser.add_argument('--max_steps', type=int, default=20)
parser.add_argument('--file_name', type=str, default='V8_test.json')
parser.add_argument('--aim_gpu', type=int, default=1)
parser.add_argument('--policy', type=str, default='Qwen3-1.7B')
parser.add_argument('--critic', type=str, default='genprm1.5B')

args = parser.parse_args()
policy_dir = args.policy
critic_dir = args.critic

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)
if '4B' in args.policy:
    policy_model = LLM(model=policy_dir, tensor_parallel_size=1,
                       max_model_len=8096*2, trust_remote_code=True, gpu_memory_utilization=0.5)
    critic_model = LLM(model=critic_dir, tensor_parallel_size=1,
                       max_model_len=8096*2, trust_remote_code=True, gpu_memory_utilization=0.9)
else:
    policy_model = LLM(model=policy_dir, tensor_parallel_size=1,
                       max_model_len=8096*2, trust_remote_code=True, gpu_memory_utilization=0.4)
    critic_model = LLM(model=critic_dir, tensor_parallel_size=1,
                       max_model_len=8096*2, trust_remote_code=True, gpu_memory_utilization=0.9)
policy_tokenizer = AutoTokenizer.from_pretrained(
    policy_dir, trust_remote_code=True)
critic_tokenizer = AutoTokenizer.from_pretrained(
    critic_dir, trust_remote_code=True)
if not policy_tokenizer.pad_token:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
policy_stop_token = policy_tokenizer.eos_token

if not critic_tokenizer.pad_token:
    critic_tokenizer.pad_token = critic_tokenizer.eos_token
critic_stop_token = critic_tokenizer.eos_token

if 'math' in data_path.lower():
    return ''
elif 'aime' in data_path.lower():
    return 'You are a helpful math assistant. '
elif 'amc' in args.data_path.lower():
    system_prompt = 'You are a helpful math assistant.'
elif 'reclor' in args.data_path.lower() or 'gpqa' in args.data_path.lower() or 'logiqa' in args.data_path.lower():
    system_prompt = 'You are a helpful assistant. Here is a question and four candidate answers. You need to reason step by step and choose the most likely answer from the four candidate answers. Answer "A", "B", "C", or "D".'
elif 'strategyqa' in args.data_path.lower():
    system_prompt = 'You are a helpful assistant. After each step, you may receive a feedback from the user, indicating that the previous step is incorrect. You should then revise your solution accordingly. Please answer "Yes" or "No".'

DATA_PATH = args.data_path
with open(DATA_PATH, encoding='utf-8') as file:
    test_data = json.load(file)
all_res = []
start_time = time.time()
all_policy_output_tokens = 0
all_critic_output_tokens = 0
for test_data_idx in range(len(test_data)):
    print(f"Processing {test_data_idx} / {len(test_data)}")
    tem_test_data = test_data[test_data_idx]
    momentum_uncertainty = 0
    current_traj = []
    max_steps = args.max_steps
    get_answer = False
    question = tem_test_data['input']
    for step_idx in range(max_steps):
        try:
            question = tem_test_data['input']

            policy_inputs = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': 'Q: ' + question +
                    "\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step_idx}:' " + '\n'},
            ]

            inputs = policy_tokenizer.apply_chat_template(
                policy_inputs,
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True
            )
            if step_idx > 0:
                inputs = inputs.replace(policy_stop_token, "").strip(
                ) + '\n'.join(current_traj) + '\n' + f'Step{step_idx}:'
            else:
                inputs = inputs.replace(
                    policy_stop_token, "").strip() + f'\nStep0:'
            sampling_params = SamplingParams(
                max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1)

            outputs = policy_model.generate(inputs, sampling_params)
            all_policy_output_tokens += len(outputs[0].outputs[0].token_ids)
            output = outputs[0].outputs[0]
            logp = output.cumulative_logprob
            all_avglogp = logp/(len(output.token_ids)+1e-8)
            cur_signal = all_avglogp
            current_traj.append('Step' + str(step_idx) +
                                ': ' + output.text.strip())
            if "the answer is" in ''.join(current_traj).lower():
                get_answer = True
                break

            if np.exp(cur_signal) < np.exp(momentum_uncertainty)*args.scaling_rate and output.text.strip() != '' and step_idx > 0:
                print('cirtic question ', test_data_idx, '  step_idx', step_idx)
                if 'math' in args.data_path.lower() or 'aime' in args.data_path.lower():
                    critic_prompt_dict = ciritique_last_generation_math(
                        question, current_traj)
                else:
                    critic_prompt_dict = ciritique_last_generation(
                        question, current_traj)
                critic_inputs = critic_tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': critic_prompt_dict['system_prompt']},
                     {'role': 'user',
                         'content': critic_prompt_dict['user_prompt']},
                     ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                critic_inputs = critic_inputs.replace(
                    critic_stop_token, "").strip()

                analyze_template = "<analyze>\nLet's analyze the paragraph {cur_step} step by step: "
                analyze_start = analyze_template.format(cur_step=step_idx)
                critic_inputs = critic_inputs + analyze_start
                sampling_params = SamplingParams(max_tokens=4096, temperature=0.6, stop=[
                                                 '</analyze>\n', '```python'], include_stop_str_in_output=True)
                critic_outputs = critic_model.generate(
                    critic_inputs, sampling_params)
                analyze_text = critic_outputs[0].outputs[0].text.strip()
                all_critic_output_tokens += len(
                    critic_outputs[0].outputs[0].token_ids)
                output_start = "<output>\n**Judgement**: $\\boxed"
                critic_inputs = critic_inputs + analyze_text + output_start
                sampling_params = SamplingParams(max_tokens=4096, temperature=0.6, stop=[
                                                 '</output>\n', '</think>\n', '```python'], include_stop_str_in_output=True)
                critic_outputs = critic_model.generate(
                    critic_inputs, sampling_params)
                output_text = critic_outputs[0].outputs[0].text.strip()
                all_critic_output_tokens += len(
                    critic_outputs[0].outputs[0].token_ids)
                critic = "<analyze>" + analyze_text + output_start + output_text

                analyze_text = re.search(
                    r'<analyze>(.*?)</analyze>', critic, re.DOTALL).group(1)
                if '</think>' in critic:
                    output_text = re.search(
                        r'<output>(.*?)</think>', critic, re.DOTALL).group(1)
                else:
                    output_text = re.search(
                        r'<output>(.*?)</output>', critic, re.DOTALL).group(1)
                analyze_text = analyze_text.replace('<analyze>', '').replace(
                    '</analyze>', '').replace('paragraph_', 'Step').replace('paragraph', 'Step')
                analyze_text = analyze_text.replace(
                    '**Judgement**:', 'So the correctness of the step is:')
                if not 'yes' in output_text.lower():
                    question = tem_test_data['input']
                    policy_inputs = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'Q: ' + question +
                            "\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step_idx}:' " + '\n'},
                        {'role': 'assistant',
                            'content':  '\n'.join(current_traj)},
                        {'role': 'user', 'content': f"\nYour previous solution is incorrect.\n{analyze_text}\nPlease revise your solution."},
                        {'role': 'assistant', 'content': 'Refined Step' +
                            str(step_idx) + ': '}
                    ]

                    inputs = policy_tokenizer.apply_chat_template(
                        policy_inputs,
                        tokenize=False,
                        enable_thinking=False
                    )
                    inputs = inputs.replace(policy_stop_token, "").strip()
                    sampling_params = SamplingParams(max_tokens=2048, temperature=0.6, stop=[
                                                     "Step"], logprobs=1)

                    outputs = policy_model.generate(inputs, sampling_params)
                    all_policy_output_tokens += len(
                        outputs[0].outputs[0].token_ids)
                    output = outputs[0].outputs[0]
                    cur_signal = output.cumulative_logprob / \
                        (len(output.token_ids)+1e-8)
                    current_traj[-1] = 'Step' + \
                        str(step_idx) + ': ' + output.text.strip()
        except:
            pass

        momentum_uncertainty = args.momentum_rate * \
            momentum_uncertainty + (1 - args.momentum_rate) * cur_signal

    if not get_answer:
        try:
            question = tem_test_data['input']
            policy_inputs = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': 'Q: ' + question +
                    "\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step_idx}:' " + '\n'},
            ]

            inputs = policy_tokenizer.apply_chat_template(
                policy_inputs,
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True
            )

            inputs = inputs.replace(policy_stop_token, "").strip(
            ) + '\n'.join(current_traj) + '\n' + f'Step{step_idx}:'
            if 'aime' in args.data_path:
                sampling_params = SamplingParams(
                    max_tokens=8096, temperature=0.6, logprobs=1)
            else:
                sampling_params = SamplingParams(
                    max_tokens=8096, temperature=0.6, logprobs=1)

            outputs = policy_model.generate(inputs, sampling_params)
            all_policy_output_tokens += len(outputs[0].outputs[0].token_ids)
            output = outputs[0].outputs[0]

            current_traj.append(output.text.strip())
        except:
            pass

    tem_res = {
        'question': question,
        'ground_truth': tem_test_data['target'],
        'current_traj': '\n'.join(current_traj),
        'final_answer': current_traj[-1] if current_traj else 'No answer'
    }
    all_res.append(tem_res)
    with open('res/'+args.file_name+'.json', 'w') as f:
        json.dump(all_res, f, indent=4)


end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")


with open('res/time/'+args.file_name+'.txt', 'w') as f:
    f.write('\n\n' + args.file_name + '  time: ' +
            str(end_time - start_time) + '\n\n')
    f.write('all_policy_output_tokens: ' +
            str(all_policy_output_tokens) + '\n')
    f.write('all_critic_output_tokens: ' +
            str(all_critic_output_tokens) + '\n')
