import os
import json
import numpy as np
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from utils.generate_prompts import ciritique_generation, ciritique_last_generation, ciritique_last_generation_math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/math_500_test.json')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--momentum_rate', type=float, default=0.9)
parser.add_argument('--max_steps', type=int, default=20)
parser.add_argument('--file_name', type=str, default='V3_test.json')
parser.add_argument('--candidate_num', type=int, default=4)
parser.add_argument('--verify_num', type=int, default=1,
                    help='for each candidate, verify how many times')
parser.add_argument('--scaling_rate', type=float, default=0.9,
                    help='scaling rate for momentum uncertainty')
parser.add_argument('--aim_gpu', type=int, default=1, help='aim for gpu')
parser.add_argument('--policy', type=str, default='Qwen3-4B-FP8')
parser.add_argument('--critic', type=str, default='genprm1.5B')
parser.add_argument('--cluster_num', type=int, default=2)
args = parser.parse_args()
policy_dir = args.policy
critic_dir = args.critic

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)
policy_model = LLM(model=policy_dir, tensor_parallel_size=1,
                   max_model_len=8096*2, trust_remote_code=True, gpu_memory_utilization=0.9)
policy_tokenizer = AutoTokenizer.from_pretrained(
    policy_dir, trust_remote_code=True)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


if not policy_tokenizer.pad_token:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
policy_stop_token = policy_tokenizer.eos_token

if 'math' in data_path.lower():
    return ''
elif 'aime' in data_path.lower():
    return 'You are a helpful math assistant. '
elif 'reclor' in args.data_path.lower() or 'gpqa' in args.data_path.lower() or 'logiqa' in args.data_path.lower():
    system_prompt = 'You are a helpful assistant. Please answer "A", "B", "C", or "D".'
elif 'strategyqa' in args.data_path.lower():
    system_prompt = 'You are a helpful assistant. Please answer "Yes" or "No".'

DATA_PATH = args.data_path
with open(DATA_PATH, encoding='utf-8') as file:
    test_data = json.load(file)
# test_data = test_data[:100]
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
            if step_idx > 0:
                calibrated_momentum_uncertainty = momentum_uncertainty / (1 - math.pow(args.momentum_rate, step_idx))
            else:
                calibrated_momentum_uncertainty = momentum_uncertainty
            if "the answer is" in ''.join(current_traj).lower():
                get_answer = True
                break

            if np.exp(cur_signal) < np.exp(momentum_uncertainty)*args.scaling_rate and output.text.strip() != '' and step_idx > 0:
                print('sampling step_idx: ', step_idx)
                cur_step_candidates = []
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
                    ) + '\n'.join(current_traj[:-1]) + '\n' + f'Step{step_idx}:'
                else:
                    inputs = inputs.replace(
                        policy_stop_token, "").strip() + f'\nStep0:'
                sampling_params = SamplingParams(max_tokens=2048, temperature=0.6, stop=[
                                                 "Step"], logprobs=1, n=args.candidate_num)

                outputs = policy_model.generate(inputs, sampling_params)
                all_cumulative_logp = []
                for cur_output in outputs[0].outputs:
                    cur_step_candidates.append(cur_output.text.strip())
                    all_cumulative_logp.append(
                        cur_output.cumulative_logprob/(len(cur_output.token_ids)+1e-8))
                    all_policy_output_tokens += len(cur_output.token_ids)

                foresight_inputs = []
                for forsight_idx in range(len(all_cumulative_logp)):
                    foresight_input = inputs + \
                        '\n'.join(
                            current_traj[:-1]) + '\n' + f'\nStep{step_idx}:' + cur_step_candidates[forsight_idx]
                    foresight_inputs.append(foresight_input)
                sampling_params = SamplingParams(
                    max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1)
                foresight_outputs = policy_model.generate(
                    foresight_inputs, sampling_params)

                foresight_cumulative_logp = []
                foresight_text = []
                for foresight_idx in range(len(all_cumulative_logp)):
                    foresight_cumulative_logp.append(foresight_outputs[foresight_idx].outputs[0].cumulative_logprob/(
                        len(foresight_outputs[foresight_idx].outputs[0].token_ids)+1e-8))
                    foresight_text.append(
                        foresight_outputs[foresight_idx].outputs[0].text.strip())

                renewed_foresight_text = []
                renewed_foresight_cumulative_logp = []
                for text, cumulative_logp in zip(foresight_text, foresight_cumulative_logp):
                    if text != '':
                        renewed_foresight_text.append(text)
                        renewed_foresight_cumulative_logp.append(
                            cumulative_logp)
                foresight_cumulative_logp = renewed_foresight_cumulative_logp
                foresight_text = renewed_foresight_text
                if len(foresight_text) == 0:
                    print("foresight_text is empty at step: ", step_idx)
                    continue

                try:
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(foresight_text)
                    k = args.cluster_num
                    kmeans = KMeans(n_clusters=k)
                    kmeans.fit(X)
                    cluster_labels = kmeans.labels_
                    cluster_list = [[] for _ in range(k)]
                    for aidx, cluster_label in enumerate(cluster_labels):
                        cluster_list[cluster_label].append(aidx)
                    cluster_list = [sorted(cluster)
                                    for cluster in cluster_list]

                    cluster_len_ratio = [
                        len(cluster)/len(foresight_text) for cluster in cluster_list]
                    per_sample_cluster_len_ratio = [
                        cluster_len_ratio[cluster_labels[ddi]] for ddi in range(len(foresight_text))]
                    cluster_weights = softmax(per_sample_cluster_len_ratio)

                    foresight_cumulative_logp = softmax(
                        foresight_cumulative_logp)

                    weighted_foresight_cumulative_logp = [
                        (foresight_cumulative_logp[i] + cluster_weights[i])/2 for i in range(len(foresight_text))]
                    best_cluster_idx = np.random.choice(
                        range(len(foresight_text)), p=weighted_foresight_cumulative_logp)
                    best_cluster_text = foresight_text[best_cluster_idx]
                    print('success foresight at step: ', step_idx, '  ', )
                except:
                    if len(foresight_text) != 0:
                        best_candidate_idx = np.random.choice(range(len(foresight_text)), p=np.exp(
                            foresight_cumulative_logp)/sum(np.exp(foresight_cumulative_logp)))
                        print("fail foresight at step: ", step_idx)
                    else:
                        best_candidate_idx = np.random.choice(range(len(all_cumulative_logp)), p=np.exp(
                            all_cumulative_logp)/sum(np.exp(all_cumulative_logp)))
                        print(
                            "fail foresight at step and foresight_text is empty: ", step_idx)

                best_candidate = cur_step_candidates[best_candidate_idx]
                cur_signal = all_cumulative_logp[best_candidate_idx]
                current_traj[-1] = best_candidate

            momentum_uncertainty = args.momentum_rate * \
                momentum_uncertainty + (1 - args.momentum_rate) * cur_signal

            if "the answer is" in ''.join(current_traj).lower():
                get_answer = True
                break
        except:
            pass
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
            ) + '\n'.join(current_traj) + '\n' + f'\nStep{step_idx}:'
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
    with open('res/' + args.file_name + '.json', 'w') as f:
        json.dump(all_res, f, indent=4)


end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")


with open('res/time/' + args.file_name + '.txt', 'w') as f:
    f.write('\n\n' + args.file_name + '  time: ' +
            str(end_time - start_time) + '\n\n')
    f.write('all_policy_output_tokens: ' +
            str(all_policy_output_tokens) + '\n')
    f.write('all_critic_output_tokens: ' +
            str(all_critic_output_tokens) + '\n')
