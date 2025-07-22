import os
import json
import numpy as np
import argparse
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def setup_model(model_path, gpu_memory_utilization=0.9):
    model = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8096*2,
                trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, tokenizer.eos_token


def get_system_prompt(data_path):
    path = data_path.lower()
    if any(x in path for x in ['math', 'aime']):
        return 'You are a helpful math assistant.'
    elif any(x in path for x in ['reclor', 'gpqa', 'logiqa']):
        return 'You are a helpful assistant. Please answer "A", "B", "C", or "D".'
    elif 'strategyqa' in path:
        return 'You are a helpful assistant. Please answer "Yes" or "No".'
    return ''


def build_prompt(tokenizer, question, traj, step_idx, stop_token):
    chat = [{'role': 'user', 'content': f"Q: {question}\nAlways end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step_idx}:'\n"}]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    prompt = prompt.replace(stop_token, "").strip()
    if step_idx > 0:
        prompt += '\n' + '\n'.join(traj) + f'\nStep{step_idx}:'
    else:
        prompt += '\nStep0:'
    return prompt


def foresight_rerank(policy_model, tokenizer, base_prompt, step_candidates, current_traj, step_idx, cluster_num):
    foresight_inputs = [base_prompt + '\n'.join(
        current_traj[:-1]) + f'\nStep{step_idx}:' + cand for cand in step_candidates]
    sampling_params = SamplingParams(
        max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1)
    foresight_outputs = policy_model.generate(
        foresight_inputs, sampling_params)

    foresight_texts, foresight_scores = [], []
    for output in foresight_outputs:
        text = output.outputs[0].text.strip()
        if text:
            foresight_texts.append(text)
            foresight_scores.append(
                output.outputs[0].cumulative_logprob / (len(output.outputs[0].token_ids) + 1e-8))

    if not foresight_texts:
        return None, None

    try:
        X = TfidfVectorizer().fit_transform(foresight_texts)
        kmeans = KMeans(n_clusters=cluster_num, n_init='auto').fit(X)
        labels = kmeans.labels_

        cluster_sizes = [list(labels).count(i) for i in labels]
        cluster_probs = softmax(cluster_sizes)
        foresight_probs = softmax(foresight_scores)
        combined_probs = [(foresight_probs[i] + cluster_probs[i]) /
                          2 for i in range(len(foresight_scores))]

        best_idx = np.random.choice(
            range(len(foresight_texts)), p=combined_probs)
        return best_idx, foresight_texts[best_idx]
    except Exception as e:
        print("Clustering failed:", e)
        fallback_probs = softmax(foresight_scores)
        best_idx = np.random.choice(
            range(len(foresight_scores)), p=fallback_probs)
        return best_idx, foresight_texts[best_idx]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

    policy_model, policy_tokenizer, policy_stop_token = setup_model(
        args.policy)
    system_prompt = get_system_prompt(args.data_path)

    with open(args.data_path, encoding='utf-8') as f:
        test_data = json.load(f)

    all_res = []
    total_policy_tokens = 0
    start_time = time.time()

    for idx, item in enumerate(test_data):
        print(f"Processing {idx + 1} / {len(test_data)}")
        question = item['input']
        current_traj = []
        momentum_uncertainty = 0
        get_answer = False

        for step_idx in range(args.max_steps):
            try:
                prompt = build_prompt(
                    policy_tokenizer, question, current_traj, step_idx, policy_stop_token)
                outputs = policy_model.generate(prompt, SamplingParams(
                    max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1))
                output = outputs[0].outputs[0]
                total_policy_tokens += len(output.token_ids)

                step_text = output.text.strip()
                cur_logp = output.cumulative_logprob / \
                    (len(output.token_ids) + 1e-8)
                current_traj.append(f"Step{step_idx}: {step_text}")

                if "the answer is" in ''.join(current_traj).lower():
                    get_answer = True
                    break

                if np.exp(cur_logp) < np.exp(momentum_uncertainty) * args.scaling_rate and step_idx > 0 and step_text:
                    print("Sampling due to uncertainty at step:", step_idx)

                    # Generate step candidates
                    base_prompt = build_prompt(
                        policy_tokenizer, question, current_traj[:-1], step_idx, policy_stop_token)
                    outputs = policy_model.generate(base_prompt, SamplingParams(
                        max_tokens=2048, temperature=0.6, stop=["Step"], logprobs=1, n=args.candidate_num))
                    candidates = [o.text.strip() for o in outputs[0].outputs]
                    all_logps = [
                        o.cumulative_logprob / (len(o.token_ids) + 1e-8) for o in outputs[0].outputs]
                    total_policy_tokens += sum(len(o.token_ids)
                                               for o in outputs[0].outputs)

                    best_idx, best_text = foresight_rerank(
                        policy_model, policy_tokenizer, base_prompt, candidates, current_traj, step_idx, args.cluster_num)

                    if best_idx is not None:
                        current_traj[-1] = f"Step{step_idx}: {candidates[best_idx]}"
                        cur_logp = all_logps[best_idx]
                    else:
                        print("Foresight fallback triggered.")

                momentum_uncertainty = args.momentum_rate * \
                    momentum_uncertainty + (1 - args.momentum_rate) * cur_logp

            except Exception as e:
                print(f"Error at step {step_idx}: {e}")
                continue

        # Final fallback if no answer found
        if not get_answer:
            try:
                fallback_prompt = build_prompt(
                    policy_tokenizer, question, current_traj, step_idx, policy_stop_token)
                outputs = policy_model.generate(fallback_prompt, SamplingParams(
                    max_tokens=2048, temperature=0.6, logprobs=1))
                output = outputs[0].outputs[0]
                current_traj.append(output.text.strip())
                total_policy_tokens += len(output.token_ids)
            except Exception as e:
                print("Final fallback failed:", e)

        all_res.append({
            'question': question,
            'ground_truth': item['target'],
            'current_traj': '\n'.join(current_traj),
            'final_answer': current_traj[-1] if current_traj else 'No answer'
        })

        with open(f"res/{args.file_name}.json", 'w') as f:
            json.dump(all_res, f, indent=4)

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    with open(f"res/time/{args.file_name}.txt", 'w') as f:
        f.write(f"\n{args.file_name} time: {total_time:.2f} seconds\n")
        f.write(f"Total policy output tokens: {total_policy_tokens}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/math_500_test.json')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--momentum_rate', type=float, default=0.9)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--file_name', type=str,
                        default='phi_decoding-mur.json')
    parser.add_argument('--candidate_num', type=int, default=4)
    parser.add_argument('--verify_num', type=int, default=1)
    parser.add_argument('--scaling_rate', type=float, default=0.9)
    parser.add_argument('--aim_gpu', type=int, default=1)
    parser.add_argument('--policy', type=str, default='Qwen3-4B-FP8')
    # hyperparameter for phi-decoding
    parser.add_argument('--cluster_num', type=int, default=2)
    args = parser.parse_args()

    main(args)
