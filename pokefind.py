from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.agents import load_tools, initialize_agent, create_sql_agent, AgentType, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv
import os
import streamlit as st

def main():

    load_dotenv()

    st.set_page_config(page_title="Pokemon Search")
    st.header("Search for anything Pokemon related, currently only Gen 1-8 available")

    dburi = os.getenv('DATABASE_URI')
    db = SQLDatabase.from_uri(dburi)
    llm = OpenAI(temperature=0.2)
    toolkit = SQLDatabaseToolkit(llm=llm, db=db)

    dialect = 'sqlite'
    top_k = 10

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        prefix=f'You are an agent designed to interact with an SQL database containing Pokemon from generation 1 to 8.\nGiven an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer, with any relevant values asked.\nUnless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.\nYou can order the results by a relevant column to return the most interesting examples in the database.\nNever query for all the columns from a specific table, only ask for the relevant columns given the question.\nYou have access to tools for interacting with the database.\nOnly use the below tools. Only use the information returned by the below tools to construct your final answer.\nYou MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.\n\nDO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n\nIf the question does not seem related to the database, just return "I don\'t know" as the answer.\n',
    )

    input = st.text_input(label='Enter the statistic you would like to find', max_chars=200)

    if input is not None:
        st.write(agent_executor.run(input))



if __name__ == '__main__':
    main()