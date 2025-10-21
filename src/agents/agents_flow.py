
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
import operator
import os
from langchain.chat_models import init_chat_model
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from typing import Dict, List

from src.config.settings import settings

class TopicData(BaseModel):
    """二级标签数据模型"""
    entity_level2: str = Field(description="二级标签")
    keywords: List[str] = Field(description="关键词列表")

class KGResult(BaseModel):
    """搜索结果的完整模型"""
    results: Dict[str, List[TopicData]] = Field(
        default={},
        description="以一级标签为键，二级标签(entity_level2)数据列表为值",
        example={
            "抗老抗衰": [
                {"entity_level2": "抗氧化", "keywords": ["抗氧", "去氧化", "抵御氧化"]},
                {"entity_level2": "抗老", "keywords": ["抗衰老", "延缓衰老", "逆龄"]},
                {"entity_level2": "抗糖化", "keywords": ["抗糖", "防糖化", "抗AGEs"]}
            ],
            "美白提亮": [
                {"entity_level2": "美白淡斑", "keywords": ["美白", "美白焕亮", "变的白嫩"]},
                {"entity_level2": "提亮肤色", "keywords": ["提亮", "亮肤", "焕亮"]}
            ]
        }
    )

class AnalystResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    next_action: Literal["[FINAL_RESULT]", "[REQUEST_MORE_INFO]"] = Field(
        description="下一步。如果不需要补充信息了，就返回[FINAL_RESULT]。如果需要补充信息，返回[REQUEST_MORE_INFO]"
    )
    additional_info: str = Field(description="如果需要补充信息[REQUEST_MORE_INFO]，列出需要补充的信息或查询方向")
    result: KGResult = Field(description="实体一二级标签体系结果")

parser = PydanticOutputParser(pydantic_object=AnalystResponseFormatter)
format_inst=parser.get_format_instructions()

# Define the shared state
class AgentState(TypedDict):
    industry: str
    subject: str
    query: Annotated[list, operator.add]
    messages: Annotated[list, operator.add]
    research_data: Annotated[list, operator.add]
    final_result: str
    next_agent: str
    iteration_count: int


# Initialize the LLM and Serper search tool
llm = init_chat_model(model=settings.MODEL_NAME, model_provider=settings.MODEL_PROVIDER, temperature=settings.TEMPERATURE)
llm2 = init_chat_model(model=settings.MODEL_NAME, model_provider=settings.MODEL_PROVIDER, temperature=settings.TEMPERATURE)
search = GoogleSerperAPIWrapper(k=10)

def perform_search(query: str) -> str:
    """
    Perform a web search using Serper API
    """
    try:
        results = search.results(query)

        # Extract relevant information
        search_results = []

        # Add organic results
        if "organic" in results:
            for i, result in enumerate(results["organic"][:5], 1):
                search_results.append(f"""Result {i}:
                Title: {result.get('title', 'N/A')}
                Link: {result.get('link', 'N/A')}
                Snippet: {result.get('snippet', 'N/A')}""")

        # Add knowledge graph if available
        if "knowledgeGraph" in results:
            kg = results["knowledgeGraph"]
            search_results.append(f"""Knowledge Graph:
            Title: {kg.get('title', 'N/A')}
            Description: {kg.get('description', 'N/A')}""")

        # Add answer box if available
        if "answerBox" in results:
            ab = results["answerBox"]
            search_results.append(f"""
            Answer Box:{ab.get('answer', ab.get('snippet', 'N/A'))}""")

        return "\n".join(search_results) if search_results else "No results found"

    except Exception as e:
        return f"Search error: {str(e)}"


def researcher_node(state: AgentState) -> AgentState:
    """
    Researcher agent that performs searches using Serper API
    """
    print("\n🔍 RESEARCHER AGENT ACTIVE")

    # Get the last message to understand what to research
    messages = state.get("messages", [])
    subject = state.get("subject", "")
    industry = state.get("industry", "")
    previous_query = state.get("query", "")
    previous_result = state.get("final_result", "")
    if len(previous_result)>=1:
        previous_result=previous_result[-1]

    system_prompt1 = f'''
        ## 角色：
        你是一位善用搜索工具的研究员。你需要根据用户提供的信息进行改写和调整，并用这调整好的信息进行查询。
        
        ## 任务：
        你是【{industry}】行业专家，目前在整合网络上用户可能提到的各种实体【{subject}】的一二级标签体系。请针对【{industry}】行业的实体【{subject}】，编写搜索查询句子。
        
        ## 原则：
        只返回搜索查询句子，别的都不要。'''

    response = llm.invoke([SystemMessage(content=system_prompt1)])

    query_content = response.content
    last_message_content = '还没反馈'
    # Determine search query
    if not messages or len(messages) == 0:
        # Initial research
        search_query = query_content
        print(f"第一个查询: {search_query}")
    else:
        # Follow-up research based on analyst's request
        last_message = messages[-1]
        last_message_content = last_message.content

        # Use LLM to extract specific search query from analyst's request
        query_prompt = f"""
        ## 角色：
        你是一位善用搜索工具的研究员。
        
        ## 任务：
        你是【{industry}】行业专家，目前在整合网络上用户可能提到的各种实体【{subject}】的一二级标签体系。你会针对【{industry}】行业的实体【{subject}】，编写搜索查询句子。
        根据分析师的需求反馈，请再生成1个精简的网络搜索查询句子。确保搜索查询句子的字数不超过50个字.
        
        分析师反馈: {last_message_content}
        分析师目前已整合好的标签体系: {previous_result}
        目标行业: {industry}
        目标实体: {subject}
        之前你用过的搜索查询句子：{previous_query}
        
        ## 原则：
        1、只返回搜索查询句子，别的都不要。
        2、避免使用之前你用过的搜索查询句子。搜索查询句子尽量和之前的不一样。"""

        query_response = llm.invoke([SystemMessage(content=system_prompt1),
                                     HumanMessage(content=query_prompt)])
        search_query = query_response.content.strip()
        print(f"Follow-up search query: {search_query}")

    # Perform actual web search using Serper
    print(f"Searching web for: {search_query}")
    search_results = perform_search(search_query)

    system_prompt2 = f'''
            ## 角色：
            你是一位市场调研专家和【{industry}】行业专家，目前在整合网络上用户可能提到的各种实体【{subject}】的一二级标签体系。

            ## 任务：
            1、根据搜索查询回来的信息，你的任务是把所有可能的【{industry}】行业的实体【{subject}】关键词都全部列出来。
            2、尝试对每个实体【{subject}】进行归类，每个类下也列出对应的关键词。确保每个类是互斥的。'''

    # Have LLM synthesize the search results
    synthesis_prompt = f"""请根据网络搜索结果和分析师反馈，把所有可能的【{industry}】行业的【{subject}】都全部列出来。尝试对每个实体【{subject}】进行归类，每个类下也列出对应的关键词。确保每个类是互斥的。
    
    分析师反馈：{last_message_content}
    分析师目前已整合好的标签体系: {previous_result}
    目标行业：{industry}
    目标实体：{subject}
    搜索查询句子：{search_query}
    搜索查询结果：{search_results}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt2),
        HumanMessage(content=synthesis_prompt)
    ])

    research_content = response.content
    print(f"搜索后的信息总结: {research_content[:400]}...")

    return {
        "research_data": [research_content],
        "query":[query_content],
        "messages": [AIMessage(content=f"[RESEARCHER]: {research_content}", name="researcher")],
        "next_agent": "analyst"
    }

def analyst_node(state: AgentState) -> AgentState:
    """
    Analyst agent that reviews research and writes reports or requests more info
    """
    print("\n📊 ANALYST AGENT ACTIVE")

    research_data = state.get("research_data", [])
    subject = state.get("subject", "")
    industry = state.get("industry", "")
    iteration = state.get("iteration_count", 0)
    previous_result = state.get("final_result", "")
    if len(previous_result) >= 1:
        previous_result = previous_result[-1]

    system_prompt = f'''
    ## 角色：
    你是{industry}行业专家，目前在整合和扩展【{industry}】行业的关于实体【{subject}】的标签体系。搭建时会考虑以下这两点：
    1、这些标签对{industry}的客户是非常重要的。客户会用这些标签来进行分析、研发新{industry}产品和定位{subject}的人群。
    2、在{industry}的品牌客户的公司，每个部门都负责各自产品和一个细分{subject}标签。所以，每个二级标签颗粒度需要足够细（但也要有一定的讨论量，且能出insight)，也需要确保"一级和一级"或"二级和二级"之间是互斥的。
    
    ## 任务：
    1、你会先搭建一版含一级二级的标签体系框架出来。同时也列出每个二级标签下可能的关键词。
    2、如果【你上一次整合好的标签体系】里有数据，你可以根据这个信息进行标签体系框架的迭代。
    3、然后，根据【查询研究代理】的网络搜索结果，你再一次进行标签体系框架的整合和扩展。
    4、如果信息不够，你需要反馈给【查询研究代理】，让他按照你的反馈进行下一步查询，以补全信息。
    5、确保每个一级标签之间是互斥的，和确保每个二级标签之间也是互斥的
    
    目标行业：{industry}
    目标实体：{subject}
    目前查询次数：{iteration}
    你上一次整合好的标签体系：{previous_result}
    
    ## 输出格式：
    {format_inst}
    
    ## 原则：
    1、按照以上的json格式输出。
    2、请确保最终标签有5个一级实体标签，每个一级实体标签下有起码3个二级实体标签。如果不满足这个条件，请按上述的json格式输出[REQUEST_MORE_INFO]，并在additional_info里提出需要进一步的查询方向。
    3、如果上面的需求满足了，请输出[FINAL_RESULT]，和按以上提到的”输出格式"输出整个标签体系。
    4、请确保有至少2次的查询次数。
    5、请保持合理的查询次数。如果查询次数已经超过了5，请输出[FINAL_RESULT]，并便用现有的【查询研究代理】网络搜索结果数据输出一个完整的名字和链接列表。
    '''

    chat_prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system_prompt),
                                                    AIMessage(content=f'## 【查询研究代理】网络搜索结果数据 如下：\n\n{research_data}')])
    chain = chat_prompt | llm2 | parser

    analyst_response = chain.invoke({})
    next_action = analyst_response.model_dump().get('next_action')
    additional_info = analyst_response.model_dump().get('additional_info')
    result = analyst_response.model_dump().get('result')

    print(f"分析师总结: {next_action}\n\n{additional_info}\n\n{result}...")

    # Check if analyst wants more info or is ready to report
    if next_action == "[FINAL_RESULT]" or iteration >= 5:
        # Extract report (remove the [FINAL REPORT] marker if present)
        return {
            "final_result": result,
            "messages": [AIMessage(content=f"[ANALYST]: {result}", name="analyst")],
            "next_agent": "end",
            "iteration_count": iteration + 1
        }
    else:
        # Analyst needs more info
        return {
            "final_result": result,
            "messages": [AIMessage(content=additional_info, name="analyst")],
            "next_agent": "researcher",
            "iteration_count": iteration + 1
        }


def route_agent(state: AgentState) -> Literal["researcher", "analyst", "end"]:
    """
    Route to the next agent based on state
    """
    next_agent = state.get("next_agent", "researcher")
    #
    # if next_agent == "end":
    #     return END
    return next_agent


# Build the graph
def create_research_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)

    # Set entry point
    workflow.set_entry_point("researcher")

    # Add conditional edges
    workflow.add_conditional_edges(
        "researcher",
        route_agent,
        {
            "analyst": "analyst",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "analyst",
        route_agent,
        {
            "researcher": "researcher",
            "end": END
        }
    )

    return workflow.compile()


class run_research_report():
    def run(self,subject: str,industry: str):
        """
        Run the two-agent system to research and report on a subject
        """
        print(f"\n{'=' * 60}")
        print(f"Starting Name Research on industry ({industry}) - subject ({subject})")
        print(f"{'=' * 60}")

        graph = create_research_graph()

        initial_state = {
            "industry": industry,
            "subject": subject,
            "messages": [],
            "research_data": [],
            "final_result": "",
            "next_agent": "researcher",
            "iteration_count": 0
        }

        # Run the graph
        final_state = graph.invoke(initial_state)

        print(f"\n{'=' * 60}")
        print("FINAL RESULT")
        print(f"{'=' * 60}")
        print(final_state["final_result"])
        print(f"\n{'=' * 60}")

        return final_state

    def _save_result(self, subject: str, industry: str, result: str):
        """Save the report to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{industry}_{subject}_result_{timestamp}.txt"
        filepath = settings.OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SUBJECT: {subject}\n")
            f.write(f"INDUSTRY: {industry}\n")
            f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'=' * 70}\n\n")
            f.write(result)
        print(f"✅ Result saved to: {filepath}")

    def _save_research_result(self, subject: str, industry: str, result: str):
        """Save the report to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{industry}_{subject}_research_result_{timestamp}.txt"
        filepath = settings.OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"SUBJECT: {subject}\n")
            f.write(f"INDUSTRY: {industry}\n")
            f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'=' * 70}\n\n")
            f.write(result)
        print(f"✅ Result saved to: {filepath}")

