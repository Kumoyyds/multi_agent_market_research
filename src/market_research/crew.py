# from langtrace_python_sdk import langtrace
#from dotenv import load_dotenv
# load_dotenv()
import os
import time
# lang_api_key = os.getenv('LANG_API_KEY')
# langtrace.init(api_key = lang_api_key)


from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from crewai.tools import tool
from crewai_tools import SerperDevTool, FirecrawlSearchTool, ScrapeWebsiteTool

from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

text_source = TextFileKnowledgeSource(
    file_paths=["genz.txt"]
)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# use open source huggingface model

reasoner = LLM(
    model="huggingface/Qwen/QwQ-32B"
)
# can't since it seems only support text stuff 
"""
general = LLM(
    model="huggingface/google/gemma-3-27b-it"
)
"""

general = LLM(
    model="huggingface/microsoft/phi-4"
)

big_general = LLM(
    model="huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"
)

big_big_general = LLM(
    model="huggingface/meta-llama/Llama-3.3-70B-Instruct"
)



# tools
google_search = SerperDevTool(n_results=10, country='fr')
# simple web scraping tool
simple_scrape = ScrapeWebsiteTool(website_url='https://www.technavio.com/report/cosmetics-products-market-industry-in-france-analysis')
## should wait for the input
print("\n\n:) welcome welcome, we're a market research team focusing on cosmetic products for Gen Z in France. \nhow can we help you? :)\n\n")
time.sleep(2)
product = input('enter your needed product:\n ')
query = f"find information related to {product}"
firesearch_tool = FirecrawlSearchTool(url='https://connect.in-cosmetics.com/',limit=2,query=query)




@CrewBase
class MarketResearch():
    """MarketResearch crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def market_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['market_researcher'], # type: ignore[index]
            verbose=True,
            allow_delegation=False,
            tools=[firesearch_tool, google_search, simple_scrape],
            llm = general
        )

    @agent
    def customer_insight_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_insight_analyst'], # type: ignore[index]
            verbose=True,
            allow_delegation=False,
            knowledge_sources=[text_source], # type: ignore[index]
            tools = [google_search],
            llm = general

        )
    @agent
    def product_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['product_designer'], # type: ignore[index]
            allow_delegation=True,
            verbose=True,
            llm = reasoner
            # llm = reasoner
        )
    
    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['reporter'], # type: ignore[index]
            verbose=True,
            allow_delegation=True,
            llm=big_big_general
        )
    
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def market_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['market_research_task'], # type: ignore[index]
        )
    
    @task
    def customer_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['customer_analysis_task'], # type: ignore[index]
        )
    
    @task
    def design_innovation_task(self) -> Task:
        return Task(
            config=self.tasks_config['design_innovation_task'],
            context = [self.market_research_task(), self.customer_analysis_task()]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            context = [self.market_research_task(), self.customer_analysis_task(), self.design_innovation_task()],
            output_file='report.md'
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the MarketResearch crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            # tasks=self.tasks, # Automatically created by the @task decorator
            tasks = [self.reporting_task()],
            # process=Process.hierarchical,
            process=Process.sequential,  # You can use Process.hierarchical for hierarchical execution
            verbose=True
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
