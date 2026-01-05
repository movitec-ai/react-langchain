from typing import List, Union
from dotenv import load_dotenv
#from langchain.tools import Tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, tool, Tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str

from callbacks import AgentCallbackHandler



load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Return the length of the text by counting the number of characters."""
    
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name:str)-> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool 
    raise ValueError(f"Tool with name {tool_name} not found")




if __name__ == "__main__":
    print("Hello, ReAct LangChain!")
    #print(get_text_length(text="Dog"))
    tools = [get_text_length]
    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action (DO NOT generate this - it will be provided to you)
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        IMPORTANT RULES:
        1. After "Action Input:", STOP. Do NOT generate "Observation:" - it will be provided to you.
        2. If you have received an Observation and know the answer, you MUST use this EXACT format:
           Thought: I now know the final answer
           Final Answer: [your answer here]
        3. Always include both "Thought:" and "Final Answer:" lines when providing the final answer.

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names= ", ".join([t.name for t in tools])
    )

    llm = ChatOpenAI(temperature=0, stop=["Observation:", "\nObservation:", "\n\tObservation:"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"], 
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
        } 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
    )

    user_input = "What is the length of the text DOG?"
    max_iterations = 10
    iteration = 0
    agent_step = None

    while iteration < max_iterations:
        try:
            agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
                {
                    "input": user_input, 
                    "agent_scratchpad": intermediate_steps,
                }
            )
            
            if isinstance(agent_step, AgentFinish):
                print(f"\nFinal Answer: {agent_step.return_values.get('output', 'No answer provided')}")
                break
            
            if isinstance(agent_step, AgentAction):
                tool_name = agent_step.tool
                tool_to_use = find_tool_by_name(tools, tool_name)
                tool_input = agent_step.tool_input

                print(f"\nIteration {iteration + 1}:")
                print(f"  Action: {tool_name}")
                print(f"  Action Input: {tool_input}")

                observation = tool_to_use.func(str(tool_input))
                print(f"  Observation: {observation}")
                
                intermediate_steps.append((agent_step, str(observation)))
                iteration += 1
            else:
                print("Unexpected output type")
                break
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            
            # Si el error es de parsing y tenemos una observación, intentar forzar una respuesta final
            if "Could not parse LLM output" in error_msg and intermediate_steps:
                # El LLM probablemente intentó dar la respuesta final pero sin el formato correcto
                # Intentar extraer la respuesta del mensaje de error o hacer una última llamada
                print("\nAttempting to get final answer with corrected prompt...")
                try:
                    # Hacer una llamada final con un prompt más específico
                    final_prompt = f"""Based on the following information, provide ONLY the final answer in the exact format:

Question: {user_input}

Previous steps:
{format_log_to_str(intermediate_steps)}

You MUST respond with this EXACT format:
Thought: I now know the final answer
Final Answer: [your answer here]

Now provide your response:"""
                    
                    final_response = llm.invoke(final_prompt)
                    print(f"Final response: {final_response.content}")
                    
                    # Intentar parsear la respuesta final
                    try:
                        final_parsed = ReActSingleInputOutputParser().parse(final_response.content)
                        if isinstance(final_parsed, AgentFinish):
                            print(f"\nFinal Answer: {final_parsed.return_values.get('output', 'No answer provided')}")
                            break
                    except:
                        # Si aún no se puede parsear, extraer manualmente
                        content = final_response.content
                        if "Final Answer:" in content:
                            answer = content.split("Final Answer:")[-1].strip()
                            print(f"\nFinal Answer: {answer}")
                            break
                except Exception as e2:
                    print(f"Could not recover from error: {e2}")
            
            break
    
    if iteration >= max_iterations:
        print("Maximum iterations reached")

