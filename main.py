"""
Main entry point for the LangGraph Research Agent System
"""
import os
from dotenv import load_dotenv
from src.config.settings import settings
from src.agents import run_research_report
import argparse


if __name__ == "__main__":
    # Example: Research and report on a topic
    # subject = "The impact of artificial intelligence on healthcare in 2024"
    # subject = "人工智能在对大健康领域的影响"
    # agent_system_init = run_research_report()
    # result = agent_system_init.run(subject)

    parser = argparse.ArgumentParser(description="Run research report agent")
    parser.add_argument("--subject", required=True, help="The research subject (e.g. 产品功效)")
    parser.add_argument("--industry", required=True, help="The industry (e.g. 护肤)")

    args = parser.parse_args()
    # Extract values
    subject = args.subject
    industry = args.industry

    # subject = "SK2小灯泡"
    # industry = "护肤"
    agent_system_init = run_research_report()
    result = agent_system_init.run(subject,industry)

    agent_system_init._save_result(subject,industry,str(result['final_result']))
    agent_system_init._save_research_result(subject,industry,str(result['research_data']))

    # print(result)
    # print(result.keys())
    # Access the results
    # print("\n\nConversation flow:")
    # for enum,msg in enumerate(result["messages"]):
    #     print(f'################## {enum+1} #####################')
    #     print(f"\n{msg.name.upper()}: {msg.content[:150]}...")
    #
    # for i,j in result.items():
    #     print('####################################')
    #     print(i,j)

