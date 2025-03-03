
from datetime import datetime
from .ai.providers import trim_prompt, generate_completions
from .prompt import system_prompt
import asyncio


async def write_outline(prompt, learnings_string, client, model):
    """
        Generate the outline for the deep research report."
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_prompt = f"""Write an outline for a deep research report.
Here is the format of your writing:
1. Use "#" Title" to indicate section title, "-" writing point to indicate writing plan below the section title. And not including "##" Title", "###" Title" and so on. 
2. Do not include other information.
3. Do not include topic name itself in the outline.
4. Do not include references section part in the outline.


The topic you want to write:
{prompt}

## Output Example
# xxx
- yyy
- zzz

# xxx
- yyy
- zzz

...
"""
    

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        # format=FinalReportResponse.model_json_schema(),
        # format={"type": "json_object"},
    )

    outlines = response.choices[0].message.content

    return outlines


async def write_outline_polish(prompt, learnings_string, client, model, draft_outline):
    """
        polish the outline base on the collection information"
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_prompt = f"""Improve an outline for a deep research report. You already have a draft outline that covers the general information. Now you want to improve it based on the collection information to make it more informative.
Here is the format of your writing:
1. Use "#" Title" to indicate section title, "-" writing point to indicate writing plan below the section title. And not including "##" Title", "###" Title" and so on. 
2. Do not include other information.
3. Do not include topic name itself in the outline.
4. Do not include references section part in the outline.
5. Please use the information collected to improve the draft outline.


### 注意：
不要只是罗列论点，而是将它们组织成严密、环环相扣的结构；
寻找逐层递进的因果论据，回应 "Main Topic"


The topic you want to write:
{prompt}

The collection information:
{learnings_string}

Draft outline:
{draft_outline}

## Output Example
# xxx
- yyy
- zzz

# xxx
- yyy
- zzz

...
"""
    

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        # format=FinalReportResponse.model_json_schema(),
        # format={"type": "json_object"},
    )
    outlines = response.choices[0].message.content

    return outlines


def get_first_level_section_names(outlines):
    """
        根据大纲返回结果
    """
    # 使用正则表达式匹配标题和子标题
    import re
    pattern = re.compile(r'(?P<first_subtitle>^# .+)|(?P<second_subtitle>^- .+)', re.MULTILINE)

    # 初始化变量
    result = []
    current_first_subtitle = ""
    second_subtitles = []

    # 遍历匹配结果
    for match in pattern.finditer(outlines):
        if match.group("first_subtitle"):
            # 如果遇到新的一级标题，保存之前的记录
            if current_first_subtitle and second_subtitles:
                result.append({
                    "first_subtitle": current_first_subtitle,
                    "second_subtitle": second_subtitles
                })
            # 更新当前的一级标题
            current_first_subtitle = match.group("first_subtitle").strip()
            second_subtitles = []  # 重置二级标题列表
        elif match.group("second_subtitle"):
            # 添加二级标题到当前列表
            second_subtitles.append(match.group("second_subtitle").strip())

    # 添加最后一个记录
    if current_first_subtitle and second_subtitles:
        result.append({
            "first_subtitle": current_first_subtitle,
            "second_subtitle": second_subtitles
        })
    return result

async def generate_section(prompt, learnings_string, model, client, outlines, first_subtitle, second_subtitle):
    """
        section report generate parallel
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    second_subtitle_str = ";".join(second_subtitle)
    user_prompt =  f"""Write a deep research report section based on the collected information.

Here is the format of your writing:
1. Use "#" Title" to indicate section title, don't generate a "##" Title.
2. Write the section with proper format (Start your writing with # section title. Don't include the page title or try to write other sections):
3. Please generate 3-5 paragraphs. Each paragraph is at least 1000 words long.
4. Please ensure that the data of the article is true and reliable, the logical structure is clear, the content is complete, and the style is professional, so as to attract readers to read.

The Collected information:
{learnings_string}

The topic you want to write:{prompt}

The section title you want to write:
section title:{first_subtitle}
section writing potin:{second_subtitle_str}
"""


    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        # format=FinalReportResponse.model_json_schema(),
        # format={"type": "json_object"},
    )
    section_content = response.choices[0].message.content
    return section_content


async def generate_section_serial(prompt, learnings_string, model, client, outlines, first_subtitle, second_subtitle, prev_article):
    """
        section report generate by serial 
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    second_subtitle_str = ";".join(second_subtitle)
    user_prompt = f"""Write a deep research report section based on the collected information and the already written text.

Here is the format of your writing:
1. Use "#" Title" to indicate section title, don't generate a "##" Title.
2. Write the section with proper format (Start your writing with # section title. Don't include the page title or try to write other sections):
3. Please generate 3-5 paragraphs. Each paragraph is at least 1000 words long.
4. Please ensure that the data of the article is true and reliable, the logical structure is clear, the content is complete, and the style is professional, so as to attract readers to read.
5. Maintain narrative consistency with previously written sections while avoiding content duplication. Ensure smooth transitions between sections.

The Collected information:
{learnings_string}

The topic you want to write:{prompt}

The section title you want to write:
section title:{first_subtitle}
section writing potin:{second_subtitle_str}

Already written text:
{prev_article}
"""

    print("> continue writing section ...")
    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        # format=FinalReportResponse.model_json_schema(),
        # format={"type": "json_object"},
    )
    section_content = response.choices[0].message.content
    return section_content


async def polish_article(prompt, outlines, article, model, client):
    """
        polish the article
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_prompt = f"""You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#") appropriately. Do your job for the following article.

Here is the format of your writing:
1. Use "#" Title" to indicate section title, don't generate a "##" Title.
2. Please ensure that the data of the article is true and reliable, the logical structure is clear, the content is complete, and the style is professional, so as to attract readers to read.


The topic you want to write:{prompt}

The outlines of the article:
{outlines}

The draft article:
{article}
"""

    print("> polish acticle ")    
    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        # format=FinalReportResponse.model_json_schema(),
        # format={"type": "json_object"},
    )
    full_content = response.choices[0].message.content
    return full_content
    



async def generate_article(prompt, learnings_string, client, model, outlines, writing_method="polish"):
    """
        根据section去并行生成
    """
    
    from concurrent.futures import as_completed
    import concurrent.futures
    max_thread_num = 10

    sections_to_write = get_first_level_section_names(outlines)
    section_output_dict_collection = {}

    if writing_method == "parallel" or writing_method == "polish":
        print("< parallel generate article ")
        tasks = []

        for index, section_title in enumerate(sections_to_write):
            first_subtitle = section_title['first_subtitle']
            second_subtitle = section_title['second_subtitle']

            tasks.append(generate_section(prompt, learnings_string, model, client, outlines, first_subtitle, second_subtitle))
            # tasks.append((task, index))
        
        tasks = await asyncio.gather(*tasks)
    
        # 收集结果，保持顺序
        # for task, index in tasks:
        #     section_output_dict_collection[index] = task.result().strip()
        # sorted_dict = dict(sorted(section_output_dict_collection.items(), key=lambda item: item[0]))
        
        article = "\n\n".join(tasks)

        # polish the article
        if writing_method == "polish":
            article = await polish_article(prompt, outlines, article, model, client)
    elif writing_method == "serial":
        # serial generate the article
        prev_article = ""
        print("< serial generate article ")
        article = ""
        for section_title in sections_to_write:
            first_subtitle = section_title['first_subtitle']
            second_subtitle = section_title['second_subtitle']


            section_content = await generate_section_serial(prompt, learnings_string, model, client, outlines, first_subtitle, second_subtitle, prev_article)

            article += section_content
            prev_article = section_content
    return article