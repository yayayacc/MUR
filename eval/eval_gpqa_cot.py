import json


def getAnswer(response):
    if "\\boxed" in response[-50:]:
        pred = response.split("\\boxed")[-1]
    else:
        pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char

    return ""


prediction = []
with open(f"file_path") as file:
    prediction = json.load(file)
print(len(prediction))
correct_num = 0
for i in range(len(prediction)):
    if "final_answer" in prediction[i]:
        response = prediction[i]['final_answer']
        pred = getAnswer(response)
        gt = prediction[i]['ground_truth']
        try:
            if gt == pred:
                correct_num += 1
        except:
            continue
    elif "all_answers" in prediction[i]:
        response = prediction[i]
        gt = prediction[i]['ground_truth']

        pred = getAnswer(response['all_answers'][0])
        if pred == gt:
            correct_num += 1
    else:
        response = prediction[i]
        gt = prediction[i]['ground_truth']
        if "all_answers" in response:
            all_ans = [0, 0, 0, 0]
            for each in response["all_answers"]:
                char = getAnswer(each)
                if char == "A":
                    all_ans[0] += 1
                elif char == "B":
                    all_ans[1] += 1
                elif char == "C":
                    all_ans[2] += 1
                elif char == "D":
                    all_ans[3] += 1
                if gt == "A":
                    num_gt = 0
                elif gt == "B":
                    num_gt = 1
                elif gt == "C":
                    num_gt = 2
                elif gt == "D":
                    num_gt = 3
            max_ans = max(all_ans)
            if all_ans.index(max_ans) == num_gt:
                correct_num += 1
print(correct_num/len(prediction))
