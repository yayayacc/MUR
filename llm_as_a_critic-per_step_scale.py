# llm as critic per step
import os
import json
import numpy as np
import re
import argparse
import time

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils.generate_prompts import ciritique_last_generation, ciritique_last_generation_math


def setup_model_and_tokenizer(model_path, gpu_mem):
    model = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8096*2,
                trust_remote_code=True, gpu_memory_utilization=gpu_mem)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, tokenizer.eos_token


def get_system_prompt(data_path):
    path = data_path.lower()
    if any(x in path for x in ['math', 'aime', 'amc']):
        return 'You are a helpful math assistant.'
    elif any(x in path for x in ['reclor', 'gpqa', 'logiqa']):
        return 'You are a helpful assistant. Here is a question and four candidate answers. You need to reason step by step and choose the most likely answer from the four candidate answers. Answer "A", "B", "C", or "D".'
    elif 'strategyqa' in path:
        return 'You are a helpful assistant. After each step, you may receive a feedback from the user, indicating that the previous step is incorrect. You should then revise your solution accordingly. Please answer "Yes" or "No".'
    return ''


def extract_critic_judgment(critic_output_text):
    try:
        analyze_text = re.search(
            r'<analyze>(.*?)</analyze>', critic_output_text, re.DOTALL).group(1)
    except:
        analyze_text = ''
    try:
        output_text = re.search(
            r'<output>(.*?)</(?:output|think)>', critic_output_text, re.DOTALL).group(1)
    except:
        output_text = ''

    analyze_text = analyze_text.replace('<analyze>', '').replace('</analyze>', '')\
        .replace('paragraph_', 'Step').replace('paragraph', 'Step')\
        .replace('**Judgement**:', 'So the correctness of the step is:')
    return analyze_text, output_text


def apply_chat(tokenizer, messages, stop_token, add_gen=True):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False, add_generation_prompt=add_gen)
    return inputs.replace(stop_token, "").strip()


def run(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

    policy_model, policy_tokenizer, policy_stop = setup_model_and_tokenizer(
        args.policy, 0.4)
    critic_model, critic_tokenizer, critic_stop = setup_model_and_tokenizer(
        args.critic, 0.9)
    system_prompt = get_system_prompt(args.data_path)

    with open(args.data_path, encoding='utf-8') as f:
        test_data = json.load(f)

    all_res = []
    total_policy_tokens = total_critic_tokens = 0
    start_time = time.time()

    for idx, item in enumerate(test_data):
        print(f"Processing {idx + 1} / {len(test_data)}")
        question = item['input']
        current_traj = []
        momentum_uncertainty = 0
        get_answer = False

        for step in range(args.max_steps):
            try:
                prompt = apply_chat(policy_tokenizer, [
                    {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"}
                ], policy_stop)
                if step > 0:
                    prompt += '\n' + '\n'.join(current_traj) + f'\nStep{step}:'
                else:
                    prompt += '\nStep0:'

                outputs = policy_model.generate(prompt, SamplingParams(
                    max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1))
                output = outputs[0].outputs[0]
                step_text = output.text.strip()
                logp = output.cumulative_logprob
                total_policy_tokens += len(output.token_ids)

                avg_logp = logp / (len(output.token_ids) + 1e-8)
                current_traj.append(f"Step{step}: {step_text}")

                if "the answer is" in ''.join(current_traj).lower():
                    get_answer = True
                    break

                # Trigger critic if uncertainty exceeds threshold
                if np.exp(avg_logp) < np.exp(momentum_uncertainty) * args.scaling_rate and step > 0 and step_text:
                    print(f"Critic invoked for question {idx}, step {step}")
                    if 'math' in args.data_path.lower() or 'aime' in args.data_path.lower():
                        prompt_dict = ciritique_last_generation_math(
                            question, current_traj)
                    else:
                        prompt_dict = ciritique_last_generation(
                            question, current_traj)

                    critic_prompt = apply_chat(critic_tokenizer, [
                        {'role': 'system',
                            'content': prompt_dict['system_prompt']},
                        {'role': 'user', 'content': prompt_dict['user_prompt']}
                    ], critic_stop)

                    analyze_input = critic_prompt + \
                        f"\n<analyze>\nLet's analyze the paragraph {step} step by step: "
                    analyze_outputs = critic_model.generate(analyze_input, SamplingParams(
                        max_tokens=4096, temperature=0.6, stop=['</analyze>\n', '```python'], include_stop_str_in_output=True))
                    analyze_text = analyze_outputs[0].outputs[0].text.strip()
                    total_critic_tokens += len(
                        analyze_outputs[0].outputs[0].token_ids)

                    output_input = analyze_input + analyze_text + \
                        "\n<output>\n**Judgement**: $\\boxed"
                    judge_outputs = critic_model.generate(output_input, SamplingParams(max_tokens=4096, temperature=0.6, stop=[
                                                          '</output>\n', '</think>\n', '```python'], include_stop_str_in_output=True))
                    output_text = judge_outputs[0].outputs[0].text.strip()
                    total_critic_tokens += len(
                        judge_outputs[0].outputs[0].token_ids)

                    analyze_content, judge_content = extract_critic_judgment(
                        f"<analyze>{analyze_text}<output>{output_text}")

                    if "yes" not in judge_content.lower():
                        revision_prompt = apply_chat(policy_tokenizer, [
                            {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"},
                            {'role': 'assistant',
                                'content': '\n'.join(current_traj)},
                            {'role': 'user', 'content': f"\nYour previous solution is incorrect.\n{analyze_content}\nPlease revise your solution."},
                            {'role': 'assistant', 'content': f"Refined Step{step}: "}
                        ], policy_stop, add_gen=False)

                        outputs = policy_model.generate(revision_prompt, SamplingParams(
                            max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1))
                        revised_text = outputs[0].outputs[0].text.strip()
                        total_policy_tokens += len(
                            outputs[0].outputs[0].token_ids)
                        avg_logp = outputs[0].outputs[0].cumulative_logprob / \
                            (len(outputs[0].outputs[0].token_ids) + 1e-8)
                        current_traj[-1] = f"Step{step}: {revised_text}"

                momentum_uncertainty = args.momentum_rate * \
                    momentum_uncertainty + (1 - args.momentum_rate) * avg_logp
            except Exception as e:
                print(f"Step {step} error: {e}")

        if not get_answer:
            try:
                final_prompt = apply_chat(policy_tokenizer, [
                    {'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step}:'\n"}
                ], policy_stop) + '\n' + '\n'.join(current_traj) + f'\nStep{step}:'
                outputs = policy_model.generate(final_prompt, SamplingParams(
                    max_tokens=8096, temperature=0.6, logprobs=1))
                current_traj.append(outputs[0].outputs[0].text.strip())
                total_policy_tokens += len(outputs[0].outputs[0].token_ids)
            except Exception as e:
                print(f"Final answer fallback error: {e}")

        all_res.append({
            'question': question,
            'ground_truth': item['target'],
            'current_traj': '\n'.join(current_traj),
            'final_answer': current_traj[-1] if current_traj else 'No answer'
        })

        with open(f'res/{args.file_name}.json', 'w') as f:
            json.dump(all_res, f, indent=4)

    elapsed = time.time() - start_time
    print(f"Total time taken: {elapsed:.2f} seconds")

    with open(f'res/time/{args.file_name}.txt', 'w') as f:
        f.write(f"{args.file_name} time: {elapsed:.2f} sec\n")
        f.write(f"All policy output tokens: {total_policy_tokens}\n")
        f.write(f"All critic output tokens: {total_critic_tokens}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/gpqa_diamond_test.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--momentum_rate', type=float, default=0.9)
    parser.add_argument('--scaling_rate', type=float, default=0.9)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--file_name', type=str,
                        default='llm_as_a_critic-mur.json')
    parser.add_argument('--aim_gpu', type=int, default=1)
    parser.add_argument('--policy', type=str, default='Qwen3-1.7B')
    parser.add_argument('--critic', type=str, default='genprm1.5B')
    args = parser.parse_args()
    run(args)
