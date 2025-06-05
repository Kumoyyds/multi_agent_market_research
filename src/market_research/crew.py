# from langtrace_python_sdk import langtrace
#from dotenv import load_dotenv
# load_dotenv()
import os

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
    model="huggingface/Qwen/Qwen2.5-32B-Instruct"
)

big_big_general = LLM(
    model="huggingface/meta-llama/Llama-3.3-70B-Instruct"
)



# tools
google_search = SerperDevTool(n_results=5, country='fr')
# simple web scraping tool
simple_scrape = ScrapeWebsiteTool(website_url='https://www.technavio.com/report/cosmetics-products-market-industry-in-france-analysis')
## should wait for the input
# firesearch_tool = FirecrawlSearchTool(url='https://connect.in-cosmetics.com/',limit=15,query='')




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
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            knowledge_sources = [text_source]
            # tools=[google_search]  
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True

        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MarketResearch crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
