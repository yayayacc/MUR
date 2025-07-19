# eval results for AIME (cause each question has been solved by multiple times)
import json
from vllm import LLM, SamplingParams
import argparse
from modelscope import AutoTokenizer
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='V2_test.json')
parser.add_argument('--save_name', type=str, default='test')
parser.add_argument('--aim_gpu', type=int, default=0)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.aim_gpu)

# Replace with your model path
model_path = 'general_verifier'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LLM(model=model_path, tensor_parallel_size=1,  max_model_len=4096,trust_remote_code=True, gpu_memory_utilization=0.9)

test_path = args.test_file
test_data = []
with open(test_path, 'r') as f:
    test_data = json.load(f)
test_res = []
all_answers = []
for data_idx, data in enumerate(test_data):
    print(f"Processing {data_idx} / {len(test_data)}")
    question = data['question']
    if 'target' in data:
        ground_truth = data['target']
    else:
        ground_truth = data['ground_truth']
    if 'all_answers' in data:
        all_answers = data['all_answers']
    elif 'final_answer' in data:
        all_answers = data['final_answer']
    student_answer = ''
    for idx, answer in enumerate(all_answers):
        if 'Answer:' in answer:
            all_answers[idx] = answer.split('Answer:')[-1]
        elif 'answer:' in answer.lower():
            all_answers[idx] = answer.lower().split('answer:')[-1]
        elif 'answer' in answer.lower():
            all_answers[idx] = answer.lower().split('answer')[-1]
        else:
            all_answers[idx] = answer
        if len(all_answers[idx]) > 500:
            all_answers[idx] = all_answers[idx][-500:]
        student_answer = all_answers[idx]



        # Create prompt
        prompt = (
            f"User: ### Question: {question}\n\n"
            f"### Ground Truth Answer: {ground_truth}\n\n"
            f"### Student Answer: {student_answer}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
        )

        try:
            sampling_params = SamplingParams(max_tokens=4096, n=1, logprobs=0, temperature=0.7)
            outputs = model.generate(prompt, sampling_params)
            output = outputs[0].outputs[0]
            if "final decision: yes" in output.text.strip().lower():
                test_res.append(1)
            else:
                test_res.append(0)
        except Exception as e:
            print(e)
            test_res.append(0)

print("accuracy: ", sum(test_res) / len(test_res))

with open('res/eval/' + args.save_name + '.txt', 'w') as f:
    f.write(f"\n\ntest_data_path: {test_path}\n")
    f.write(f"accuracy: {sum(test_res) / len(test_res)}\n")
    f.write(f"test_res: {test_res}\n")