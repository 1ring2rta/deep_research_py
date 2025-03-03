from typing import List, Optional
import openai
import ollama
import json
from datetime import datetime
from .prompt import system_prompt
from .common.logging import log_error, log_event
from .ai.providers import generate_completions
from .common.token_cunsumption import (
    parse_ollama_token_consume,
    parse_openai_token_consume,
)
from deep_research_py.utils import get_service
from pydantic import BaseModel
import re

class FeedbackResponse(BaseModel):
    questions: List[str]


async def generate_feedback(
    query: str,
    concept_results: list, 
    client: Optional[openai.OpenAI | ollama.Client],
    model: str,
    max_feedbacks: int = 5,
) -> List[str]:
    """Generates follow-up questions to clarify research direction."""

    prompt = f"""\
当前时间为{datetime.now().isoformat()}    

根据用户提供的研究主题：{query}，为了更好地理解用户的研究关注点，生成最多{max_feedbacks}个澄清性问题，以帮助明确用户更关心的研究方向。

概念：
{json.dumps(concept_results, indent=4, ensure_ascii=False)}

如果原始查询已经足够清晰全面，您可以返回一个空的'questions'字段。"""

    response = await generate_completions(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # format=FeedbackResponse.model_json_schema(),
        format={"type": "json_object"},
    )

    # Parse the JSON response
    try:
        if get_service() == "ollama":
            result = json.loads(response.message.content)
            parse_ollama_token_consume("generate_feedback", response)
        else:
            # OpenAI compatible API
            # json格式兜底
            json_response = response.choices[0].message.content
            try:
                json.loads(json_response) # 为正常json
            except:
                json_response = re.findall(r"```(?:json)?\s*(.*?)\s*```", json_response, re.DOTALL)[0]

            result = json.loads(json_response)
            
            # result = json.loads(response.choices[0].message.content.strip().strip("```json").strip("```"))
            parse_openai_token_consume("generate_feedback", response)

        log_event(
            f"Generated {len(result.get('questions', []))} feedback follow-up questions for query: {query}"
        )
        log_event(f"Got feedback follow-up questions: {result.get('questions', [])}")
        return result.get("questions", [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        log_error(f"Failed to parse JSON response for query: {query}")
        return []
