from typing import List, Dict, TypedDict, Optional
import asyncio
import os
import openai
from firecrawl import FirecrawlApp
from .ai.providers import trim_prompt, generate_completions
from .prompt import system_prompt
from .common.logging import log_event, log_error
from .common.token_cunsumption import (
    parse_ollama_token_consume,
    parse_openai_token_consume,
)
from .utils import get_service
import json
from pydantic import BaseModel
import requests
import httpx
from datetime import datetime
import re

class SearchResponse(TypedDict):
    data: List[Dict[str, str]]


class SerpQuery(BaseModel):
    query: str
    research_goal: str


def bing_search(query):
    url = 'https://tgenerator.aicubes.cn/iwc-index-search-engine/search_engine/v1/search'
    
    params = {
        'query': query,
        # 'se': 'BAIDU',
        'se': 'BING',
        'limit': 5,
        'user_id': 'test',
        'app_id': 'test',
        'trace_id': 'test',
        'with_content': True
    }

    header = {
        'X-Arsenal-Auth': 'arsenal-tools'
    }
    try:
        response_dic = requests.post(url, data=params, headers=header)
        # async with httpx.AsyncClient() as client:
        #     response_dic = await client.post(url, data=params, headers=header)

        if response_dic.status_code == 200:
            response =  json.loads(response_dic.text)['data']

            # 替换为serapi googlesearch的格式

            organic_results_lst = []
            for idx, t in enumerate(response):
                position = idx +1
                title = t['title'] if t['title'] else ""
                link = t['url']
                snippet = t['summary'] if t['summary'] else ""
                date = t['publish_time'] if t['publish_time'] else ""
                source = t['data_source'] if t['data_source'] else ""
                content = t['content'] if t['content'] else ""


                if date:
                    dt_object = datetime.fromtimestamp(date)
                    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
                    date = formatted_time
                    

                organic_results_lst.append({
                    "position": position,
                    "title": title,
                    "url": link,
                    "snippet": snippet,
                    "date": date,
                    "source": source,
                    "content": content
                })
            
            # res = {
            #     "search_parameters": response_dic.json()['header'],
            #     "organic_results": organic_results_lst
            # }

            return organic_results_lst

        else:
            print(f"搜索失败，状态码：{response.status_code}")
            return []
    except Exception as e:
        print(f"请求发生错误：{str(e)}")
        return []  # 出现异常时也返回空列表

class Firecrawl:
    """Simple wrapper for Firecrawl SDK."""

    def __init__(self, api_key: str = "", api_url: Optional[str] = None):
        self.app = FirecrawlApp(api_key=api_key, api_url=api_url)

    async def search(
        self, query: str, timeout: int = 15000, limit: int = 5
    ) -> SearchResponse:
        """Search using Firecrawl SDK in a thread pool to keep it async."""
        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: bing_search(
                    query=query,
                ),
            )
            # response = await bing_search(query)

            # Handle the response format from the SDK
            if isinstance(response, dict) and "data" in response:
                # Response is already in the right format
                return response
            elif isinstance(response, dict) and "success" in response:
                # Response is in the documented format
                return {"data": response.get("data", [])}
            elif isinstance(response, list):
                # Response is a list of results
                formatted_data = []
                for item in response:
                    if isinstance(item, dict):
                        formatted_data.append(item)
                    else:
                        # Handle non-dict items (like objects)
                        formatted_data.append(
                            {
                                "url": getattr(item, "url", ""),
                                "markdown": getattr(item, "markdown", "")
                                or getattr(item, "content", ""),
                                "title": getattr(item, "title", "")
                                or getattr(item, "metadata", {}).get("title", ""),
                            }
                        )
                return {"data": formatted_data}
            else:
                print(f"Unexpected response format from Firecrawl: {type(response)}")
                return {"data": []}

        except Exception as e:
            print(f"Error searching with Firecrawl: {e}")
            print(
                f"Response type: {type(response) if 'response' in locals() else 'N/A'}"
            )
            return {"data": []}


# Initialize Firecrawl
firecrawl = Firecrawl(
    api_key=os.environ.get("FIRECRAWL_API_KEY", ""),
    api_url=os.environ.get("FIRECRAWL_BASE_URL"),
)


class SerpQueryResponse(BaseModel):
    queries: List[SerpQuery]


async def generate_serp_queries(
    parent_query: str,
    research_tree: str,
    client: openai.OpenAI,
    model: str,
    num_queries: int = 3,
) -> List[SerpQuery]:
    """生成搜索查询时保持原始目标"""
    
    prompt = f"""\
当前时间为{datetime.now().isoformat()}
    
- 研究脉络图（当前位置由 ★ 标出）：
{research_tree}

- 当前阶段目标：{parent_query}
- 确保每个问题都服务于原始研究目标

根据用户提供的研究主题，生成一组用于搜索的查询。返回一个包含 {num_queries} 个查询的 JSON 对象，对象中需包含一个 'queries' 数组字段。每个查询对象应包含 'query'（查询内容）和 'research_goal'（研究目标）字段。注意：
1. 'query' 必须是一个简单的问题，能够通过搜索引擎直接回答。
2. 确保每个查询都是唯一的，且与其他查询不相似。
3. 所有查询必须服务于原始研究目标。

示例：
- 原始目标：量子计算的商业应用现状
- 优秀问题：IBM量子计算机的最新商业合作案例
- 劣质问题：量子力学的基本原理（偏离应用方向）

用户提供的主题：<prompt>{parent_query}</prompt>"""

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        # format=SerpQueryResponse.model_json_schema(),
        format={"type": "json_object"},
    )

    try:
        if get_service() == "ollama":
            result = SerpQueryResponse.model_validate_json(response.message.content)
            parse_ollama_token_consume("generate_serp_queries", response)
        else:
            # json格式兜底
            json_response = response.choices[0].message.content
            try:
                json.loads(json_response) # 为正常json
            except:
                json_response = re.findall(r"```(?:json)?\s*(.*?)\s*```", json_response, re.DOTALL)[0]

            result = SerpQueryResponse.model_validate_json(
                json_response
            )
            parse_openai_token_consume("generate_serp_queries", response)

        queries = result.queries if result.queries else []
        log_event(f"Generated {len(queries)} SERP queries for research query: {parent_query}")
        log_event(f"Got queries: {queries}")
        return queries[:num_queries]
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        log_error(
            f"Failed to parse JSON response for query: {parent_query}, raw response: {response.choices[0].message.content}"
        )
        return []


class SerpResultResponse(BaseModel):
    learnings: List[str]
    followup_questions: List[str]


async def process_serp_result(
    query: str,
    research_tree: str,
    search_result: SearchResponse,
    client: openai.OpenAI,
    model: str,
    num_learnings: int = 3,
    num_followup_questions: int = 3,
) -> Dict[str, List[str]]:
    """Process search results to extract learnings and follow-up questions."""

    contents = [
        trim_prompt(item.get("markdown", ""), 25_000)
        for item in search_result["data"]
        if item.get("markdown")
    ]

    # Create the contents string separately
    contents_str = "".join(f"<content>\n{content}\n</content>" for content in contents)

    prompt = f"""\
当前时间为{datetime.now().isoformat()}

当前研究目标：{query}

研究脉络图（当前位置由 ★ 标出）：
{research_tree}

请基于以下内容完成以下任务：
1. 生成最多 {num_learnings} 条研究认知（learnings），要求：
   - 认知内容必须与原始研究目标直接相关
   - 仔细、严格检视内容与原始研究目标之间的因果关系，若无关你应当返回空的“learnings”和“followup_questions”
   - 每条认知应简洁明了，包含具体信息（如数据、事实、案例等）
   - 避免重复或冗余信息

2. 生成最多 {num_followup_questions} 个后续研究问题（follow-up questions），要求：
   - 问题必须直接服务于原始研究目标
   - 问题应简洁明确，能够通过搜索引擎直接回答
   - 避免生成与当前内容无关或偏离原始目标的问题

3. 返回一个 JSON 对象，包含以下字段：
   - "learnings"：认知列表（数组，每条认知为一个字符串）
   - "followup_questions"：后续问题列表（数组，每个问题为一个字符串）

内容：
{contents_str}"""    

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        # format=SerpResultResponse.model_json_schema(),
        format={"type": "json_object"},
    )

    try:
        if get_service() == "ollama":
            result = SerpResultResponse.model_validate_json(response.message.content)
            parse_ollama_token_consume("process_serp_result", response)
        else:

            json_response = response.choices[0].message.content
            try:
                result = json.loads(json_response)
            except:
                result = json.loads(
                    re.findall(
                        r"```(?:json)?\s*(.*?)\s*```", 
                        json_response, 
                        re.DOTALL
                    )[0]
                )
            result = SerpResultResponse(
                learnings=result['learnings'], 
                followup_questions=result['followup_questions']
            )
            # result = SerpResultResponse.model_validate_json(
            #     json_response
            # )
            parse_openai_token_consume("process_serp_result", response)

        log_event(
            f"Processed SERP results for query: {query}, found {len(result.learnings)} learnings and {len(result.followup_questions)} follow-up questions"
        )
        log_event(
            f"Got learnings: \n{result.learnings} \nand follow-up questions: \n{result.followup_questions}"
        )
        return {
            "learnings": result.learnings[:num_learnings],
            "followup_questions": result.followup_questions[:num_followup_questions],
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        log_error(
            f"Failed to parse SERP results for query: {query}, raw response: {response.choices[0].message.content}"
        )
        return {"learnings": [], "followup_questions": []}


class FinalReportResponse(BaseModel):
    reportMarkdown: str

import sys
sys.path.append('../deep_research_py')
from .gen_outline_acticle import *
async def write_final_report(
    prompt: str,
    learnings: List[str],
    # visited_urls: List[str],
    client: openai.OpenAI,
    model: str,
    writing_method="serial"
) -> str:
    """Generate final report based on all research learnings."""

    # learnings_string = trim_prompt(
    #     "\n".join([f"<learning>\n{learning}\n</learning>" for learning in learnings]),
    #     150_000,
    # )
    learnings_string = learnings

    # step1: 生成outline
    draft_outlines = await write_outline(prompt, learnings_string, client, model)
    print(
        f"gen draft outlines: {draft_outlines}"
    )
    log_event(f"gen draft outlines: {draft_outlines}")

    # # step2: 润色outline
    outlines = await write_outline_polish(prompt, learnings_string, client, model, draft_outlines)
    print(
        f"gen polish outlines: {outlines}"
    )
    log_event(f"gen polish outlines: {outlines}")
    
    # # step3: 生成文章
    report =  await generate_article(prompt, learnings_string, client, model, outlines, writing_method)
    

    try:
        # if get_service() == "ollama":
        #     result = FinalReportResponse.model_validate_json(response.message.content)
        #     parse_ollama_token_consume("write_final_report", response)
        # else:
        #     result = FinalReportResponse.model_validate_json(
        #         response.choices[0].message.content
        #     )
        #     parse_openai_token_consume("write_final_report", response)

        # report = result.reportMarkdown if result.reportMarkdown else ""
        # log_event(
        #     f"Generated final report based on {len(learnings)} learnings from {len(visited_urls)} sources"
        # )
        # Append sources
        # urls_section = "\n\n## Sources\n\n" + "\n".join(
        #     [f"- {url}" for url in visited_urls]
        # )
        # return report + urls_section

        return report
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        # print(f"Raw response: {response.choices[0].message.content}")
        log_error(
            f"Failed to generate final report for research query, raw response:"
        )
        return "Error generating report"



from typing import TypedDict, List, Dict, Optional
from uuid import uuid4

class ResearchNode(TypedDict):
    id: str
    query: str
    research_goal: str
    learnings: List[str]
    childs: List['ResearchNode']
    urls: List[str]
    depth: int


async def deep_research(
    breadth: int,
    depth: int,
    original_query: str,
    model: str,
    client: openai.OpenAI,
    concurrency: int,
    parent: Optional[ResearchNode] = None,
    ancestor: Optional[ResearchNode] = None,
) -> ResearchNode:
    """
    返回包含树状结构的ResearchNode
    """

    # 生成搜索查询
    serp_queries = await generate_serp_queries(
        parent_query=parent['query'],
        research_tree=generate_tree_diagram(ancestor=ancestor, current_node=parent), 
        client=client,
        model=model,
        num_queries=breadth,
    )

    async def process_query(serp_query: SerpQuery) -> ResearchNode:
        try:
            current_node: ResearchNode = {
                "id": str(uuid4()),
                "depth": depth-1,
                "query": serp_query.query,
                "research_goal": serp_query.research_goal,
                "learnings": [],
                "childs": [],
                "urls": [],
            }

            if parent:
                parent["childs"].append(current_node)

            result = await firecrawl.search(serp_query.query)
            new_urls = [item.get("url") for item in result["data"] if item.get("url")]
            
            # 处理搜索结果
            processed = await process_serp_result(
                query=serp_query.query,
                research_tree=generate_tree_diagram(
                    ancestor=ancestor, 
                    current_node=parent
                ), 
                search_result=result,
                client=client,
                model=model,
            )

            # 更新当前节点
            current_node["learnings"].extend(processed["learnings"])
            current_node["urls"].extend(new_urls)
            current_node["learnings"] = list(set(current_node["learnings"]))
            current_node["urls"] = list(set(current_node["urls"]))

            # 递归处理后续问题
            if depth > 1:
                return await deep_research(
                    breadth=max(1, breadth//2),
                    depth=depth-1,
                    concurrency=concurrency,
                    client=client,
                    model=model,
                    original_query=original_query,
                    parent=current_node,
                    ancestor=ancestor
                )
            return current_node
        except Exception as e:
            print(f"Query processing failed: {str(e)}")
            return current_node

    # 并行处理所有查询
    print(generate_tree_diagram(ancestor))
    await asyncio.gather(*[process_query(q) for q in serp_queries])
    return parent


def generate_tree_diagram(ancestor: ResearchNode, current_node: Optional[ResearchNode] = None) -> str:
    """生成带URL编号标注的研究路径图谱"""
    # 预收集所有唯一URL并建立编号映射
    url_registry = {}
    
    def collect_urls(node: ResearchNode):
        for url in node['urls']:
            if url not in url_registry:
                url_registry[url] = len(url_registry) + 1
        for child in node['childs']:
            collect_urls(child)
    
    collect_urls(ancestor)  # 构建URL注册表

    def build_tree_lines(
        node: ResearchNode,
        prefix: str = "",
        is_last: bool = True,
        is_root: bool = True
    ) -> List[str]:
        lines = []
        
        # 节点标题行
        current_marker = " ★" if node is current_node else ""
        depth_tag = f"[D{node['depth']+1}]" if not is_root else "[Origin]"
        connector = "└── " if is_last else "├── "
        main_line = f"{prefix}{connector}{depth_tag} {node['query']}{current_marker}"
        lines.append(main_line)
        
        # 研究目标区块
        goal_prefix = prefix + ("    " if is_last else "│   ")
        lines.append(f"{goal_prefix}├○ 研究目标: {trim_prompt(node['research_goal'], 60)}")
        
        # 关键认知区块
        learn_prefix = goal_prefix + "│   "
        if node['learnings']:
            lines.append(f"{learn_prefix}├• 关键认知")
            for i, learning in enumerate(node['learnings']):
                bullet = "└> " if i == len(node['learnings'])-1 else "├> "
                lines.append(f"{learn_prefix}│   {bullet}{trim_prompt(learning, 80)}")
        
        # 数据来源区块
        url_prefix = learn_prefix if node['learnings'] else goal_prefix
        if node['urls']:
            lines.append(f"{url_prefix}├◈ 数据来源")
            for i, url in enumerate(node['urls']):
                link_symbol = "└→ " if i == len(node['urls'])-1 else "├→ "
                citation_tag = f"[{url_registry[url]}] "
                lines.append(f"{url_prefix}│   {link_symbol}{citation_tag}{trim_prompt(url, 60)}")
        
        # 递归处理子节点
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node['childs']):
            is_last_child = i == len(node['childs'])-1
            lines.extend(build_tree_lines(child, child_prefix, is_last_child, False))
        
        return lines

    tree_lines = build_tree_lines(ancestor, is_root=True)
    return "\n".join([
        "研究路径图谱".center(80, "═"),
        *tree_lines,
        "═"*80,
        f"图谱统计｜认知节点: {count_nodes(ancestor)} | 唯一数据源: {len(url_registry)} | 最大深度: {get_max_depth(ancestor)}",
        "═"*80
    ])

def generate_citation_list(ancestor: ResearchNode) -> List[str]:
    """生成带编号的引用列表"""
    citations = []
    url_registry = {}
    
    def _collect(node: ResearchNode):
        for url in node['urls']:
            if url not in url_registry:
                url_registry[url] = len(url_registry) + 1
                citations.append(f"[{url_registry[url]}] {url}")
        for child in node['childs']:
            _collect(child)
    
    _collect(ancestor)
    return citations

def count_nodes(node: ResearchNode) -> int:
    return 1 + sum(count_nodes(child) for child in node['childs'])

def get_max_depth(node: ResearchNode) -> int:
    if not node['childs']:
        return node['depth']
    return max(get_max_depth(child) for child in node['childs'])
