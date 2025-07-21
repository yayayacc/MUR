def cot_rationale_generation(problem, solution):
    system_prompt = 'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'
    
    user_prompt = f"""The following is the math problem and a solution (split into paragraphs, enclosed with tags and indexed from 1):

    [Math Problem]

    {problem}

    [Solution]

    """
    

    solution = solution.split('\n\n')
    for i, para in enumerate(solution):
        user_prompt += f"<paragraph_{i+1}>\n{para}\n</paragraph_{i+1}>\n\n"

    user_prompt += "Your task is to verify the correctness of paragraph in the solution. Split your verification by '### Paragraph {{ID}}'"

    user_prompt += "Your verification for each paragraph should be wrapped by '<analyze></analyze>', you need to analyze the reasoning process and explain why the paragraph is correct or incorrect in detail." 

    user_prompt += """After all verifications, if you identify an error in a paragraph, return the **index of the paragraph where the earliest error occurs**. Otherwise, return the **index of -1 (which typically denotes "not found")**. Please put your final answer (i.e., the index) within box in the form of '$\\boxed{{INDEX}}$'."""

    return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

def ciritique_generation(problem, solution):
    system_prompt = 'You are a math teacher. Your task is to review and critique the solution.'

    user_prompt = f"""The following is a math problem and my solution. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should not solve the problem and give the final answer. All of your task is to critique. Output your judgement of whether the paragraph is correct in the form of '\\boxed{{Yes|No}}' at the end of each paragraph verification:

    [Math Problem]

    {problem}

    [Solution]

    """

    solution = solution.split('\n')
    solution = [para for para in solution if para.strip()]
    for i, para in enumerate(solution):
        user_prompt += f"<paragraph_{i+1}>\n{para}\n</paragraph_{i+1}>\n\n"

    return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

def ciritique_last_generation_math(problem, solution):

    system_prompt = 'You are a math teacher. Your task is to review and critique the paragraphs in solution directly. Output your judgement in the format of `\\boxed{Yes}` if the paragraph is correct, or `\\boxed{No}` if the paragraph is incorrect.'

    user_prompt = f"""[Math Problem]

    {problem}

    [Solution]

    """

    if 'the reasoning steps are:' in solution[0].lower():
        solution = solution[1:]
    for i, para in enumerate(solution):
        user_prompt += f"<paragraph_{i}>\n{para}\n</paragraph_{i}>\n\n"

    return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

def ciritique_last_generation(problem, solution):
    system_prompt = 'You are a teacher. Your task is to review and critique the paragraphs in solution directly. Output your judgement in the format of `\\boxed{Yes}` if the paragraph is correct, or `\\boxed{No}` if the paragraph is incorrect.'

    user_prompt = f"""[Problem]

    {problem}

    [Solution]

    """

    if 'the reasoning steps are:' in solution[0].lower():
        solution = solution[1:]
    for i, para in enumerate(solution):
        user_prompt += f"<paragraph_{i}>\n{para}\n</paragraph_{i}>\n\n"

    return {'system_prompt': system_prompt, 'user_prompt': user_prompt}





def ciritique_last_generation_r1(problem, solution):
    system_prompt = 'You are a math teacher. Your task is to review and critique the solution.'

    user_prompt = f"""The following is a math problem and my solution. Your task is to review and critique the last paragraph in the solution. Pay attention that you should not solve the problem and give the final answer. All of your task is to critique. Output your judgement of whether the last paragraph is correct in the form of '\\boxed{{Yes|No}}', and you should also give the reason for your judgement.

    [Math Problem]

    {problem}

    [Solution]

    """
    solution = solution.split('\n')
    solution = [para for para in solution if para.strip()]
    for i, para in enumerate(solution):
        user_prompt += f"<paragraph_{i+1}>\n{para}\n</paragraph_{i+1}>\n\n"

    user_prompt += "Do not solve the problem and give the final answer. Only output your judgement of whether the last paragraph is correct in the form of '\\boxed{{Yes|No}}', and you should also give the reason for your judgement in the form of '<reason></reason>'."

    return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

def ciritique_last_generation_math_thinkprm(problem, solution):
    
    system_prompt = ''

    user_prompt = f"""You are given a math problem and a proposed step-by-step solution:

    [Math Problem]

    {problem}

    [Solution]

    """
   
    if 'the reasoning steps are:' in solution[0].lower():
        solution = solution[1:]
    for i, para in enumerate(solution):
        user_prompt += f"\n{para}"
    user_prompt += "Review and critique the last step in the proposed solution to determine whether it is correct. If the solution is incomplete, only verify the provided steps."

    return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

def ciritique_last_generation_thinkprm(problem, solution):
    system_prompt = ''

    user_prompt = f"""You are given a math problem and a proposed step-by-step solution:
    
    [Problem]

    {problem}

    [Solution]

    """
  
    if 'the reasoning steps are:' in solution[0].lower():
        solution = solution[1:]
    for i, para in enumerate(solution):
        user_prompt += f"\n{para}"
    user_prompt += "Review and critique the last step in the proposed solution to determine whether it is correct. If the solution is incomplete, only verify the provided steps."


    return {'system_prompt': system_prompt, 'user_prompt': user_prompt}
