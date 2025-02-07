from typing import Annotated, Literal
from functools import partial

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.messages import ToolMessage, AIMessage

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

##TOOLS
@tool
def get_bill_amount(bill_type: str, state: Annotated[dict, InjectedState]):
    """Use this tool to fetch the respective bill amount for a given bill_type"""
    if bill_type=="electricity":
        amount = 100
    elif bill_type=="rent":
        amount = 200
    return {"msg":f"{bill_type} bill amount : {amount}", "balance": state['balance']}

@tool
def pay_electricity_bill(state: Annotated[dict, InjectedState]):
    """Use this tool to pay the electricity bill"""
    return {"msg": "electricity Bill paid successfully!", "balance": state['balance']}

@tool
def deduct_amount(amount: float, state: Annotated[dict, InjectedState]):
    """Use this tool to deduct amount from the total balance. Do not invoke this before the payment is successful!"""
    total_balance = state["balance"]
    updated_balance = total_balance - amount
    return {"msg":f"New Balance : !!!{updated_balance}", "balance": updated_balance}

@tool
def pay_rent(state: Annotated[dict, InjectedState]):
    """Use this tool to pay house rent to the landlord"""
    return {"msg":"rent paid successfully!", "balance":state['balance']}

@tool
def check_balance(state: Annotated[dict, InjectedState]):
    """Use this tool to check for account balance"""
    return {"msg":f"Balance amount: {state['balance']}", "balance":state['balance']}

tools = [check_balance, pay_rent, deduct_amount, pay_electricity_bill, get_bill_amount]
tool_executor = ToolExecutor(tools)
llm = ChatOpenAI(model="gpt-4o-mini")

##BUILD GRAPH
class AgentState(MessagesState):
    balance: float

def create_team_supervisor(llm: ChatOpenAI, system_prompt) -> str:
    """An LLM-based router."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )
    return (prompt|llm)

def create_payment_agent(llm, tools, bill_type):
    """Create an agent"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a helpful personal AI assistant. "
        "Your task is to pay different bills of the user. "
        f"Specifically you are managing the following payment types:- {bill_type}. "
        "Use the provided tools to progress towards achieving a particular goal. "
        "Before paying the bill, use your tools to ensure there is sufficient balance for the payment of the bill. "
        "If you are able to successfully pay the bills, use your tools, ensure you update the overall balance amount by deducting the respective amount from the overall balance."
        "If the payment is not possible respond with PAYMENT NOT POSSIBLE, followed by a reason stated after REASON: <state reason here in one sentence>."
        "If you have completed all the payments, you have to respond with PAYMENT DONE."
        "You have access to the following tools: {tool_names}",
         ),
         MessagesPlaceholder(variable_name="messages")
    ]
)
    prompt = prompt.partial(tool_names = ", ".join(tool.name for tool in tools), bill_type=bill_type)
    return prompt | llm.bind_tools(tools)


def agent_node(state, agent, name):
    result = agent.invoke(state)
    result = AIMessage(**result.dict(exclude={"type", "name", "balance"}),
                           name=name)
    return {"messages": [result], "balance": state['balance']}


def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input={"state": state, **tool_call['args']},
        )
        tool_invocations.append(action)
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    balance = state['balance']
    tool_messages = []
    for tc, response in zip(last_message.tool_calls, responses):
        balance = response['balance']
        content = str(response['msg'])
        name = tc['name']
        tool_call_id = tc['id']
        tool_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_call_id))

    overall_msgs = messages.copy() + tool_messages
    return {"messages": overall_msgs, "balance": balance}



def initialize_graph():
    #supervisor node
    supervisor_agent = create_team_supervisor(
        llm,
        """You are a supervisor tasked with making payments for the following bill types: Electricity.
        Based on the user request, respond which bill type agent needs to be triggered.
        If you are finished making all the payments, respond with the word FINISH.
        Do not acknowledge this message with anything else apart from Electricity or FINISH."""
    )
    supervisor_node = partial(agent_node, agent=supervisor_agent, name='supervisor')
    #electricity node
    electricity_node = partial(agent_node, agent=create_payment_agent(llm, bill_type="electricity", tools=[pay_electricity_bill, get_bill_amount, check_balance, deduct_amount]), name="electricity_agent")
    #rent agent
    rent_node = partial(agent_node, agent=create_payment_agent(llm, bill_type="rent", tools=[pay_rent, get_bill_amount, check_balance, deduct_amount]), name="rent_agent")

    #routers
    def payment_agent_router(state):
        messages = state["messages"]
        last_msg = messages[-1]
        if getattr(last_msg, 'tool_calls', None):
            return "call_tool"
        elif "PAYMENT DONE" in str(last_msg.content) or "PAYMENT NOT POSSIBLE" in str(last_msg.content):
            return "__end__"
        else:
            return "continue"

    def supervisor_router(state):
        messages = state["messages"]
        last_msg = messages[-1]
        last_msg_content = str(last_msg.content)
        if "FINISH" in last_msg_content:
            print("EXITING FROM SUP.")
            return "__end__"
        elif "Electricity" in last_msg_content:
            return "electricity_agent"
        # elif "Rent" in last_msg_content:
        #     return "rent_agent"


    #define graph
    graph_builder  = StateGraph(AgentState)
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("electricity_agent", electricity_node)
    # graph_builder.add_node("rent_agent", rent_node)
    graph_builder.add_node("call_tool", call_tool)


    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_edge("call_tool", "electricity_agent")
    # graph_builder.add_edge("call_tool", "rent_agent")


    graph_builder.add_conditional_edges("electricity_agent", payment_agent_router, {"__end__": END, "call_tool": "call_tool", "continue": "electricity_agent"})
    # graph_builder.add_conditional_edges("rent_agent", payment_agent_router, {"supervisor": "supervisor", "call_tool": "call_tool","continue": "rent_agent"})
    graph_builder.add_conditional_edges("supervisor", supervisor_router, {"electricity_agent": "electricity_agent", "__end__": END})
    # graph_builder.add_conditional_edges("supervisor", supervisor_router, {"rent_agent": "rent_agent", "__end__": END})
    # graph_builder.add_conditional_edges("supervisor", supervisor_router, {"electricity_agent": "electricity_agent", "rent_agent": "rent_agent", "__end__": END})

    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer)
    return graph



