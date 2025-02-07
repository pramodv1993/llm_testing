import random
import pytest

from langchain_core.messages import ToolMessage

from data_mocker import get_valid_messages, get_node_vs_tools_mapping
from graph_builder import initialize_graph

def update_path_with_mock_tool_call_response(graph, existing_config, tool_calls_in_node, node_vs_tools, msg_mocker):
    for tool_call in tool_calls_in_node:
        tool_call_id = tool_call['id']
        tool_name = tool_call['name']
        tool_node_name = next(node_name for node_name, tools in node_vs_tools.items() if tool_name in tools)
        if not msg_mocker.get(tool_name, None):
            raise Exception(f"No mock response found for the tool : {tool_name}")
        mock_response = msg_mocker.get(tool_name)
        mock_tool_call_msg = ToolMessage(content=mock_response["msg"], id=random.randint(1, 19999), tool_call_id=tool_call_id)
        print(f"\n ---- Added mock response for the tool call '{tool_name}' for tool-node '{tool_node_name}'---\n")
        print("Mocked Tool Response:   ", mock_tool_call_msg)
        existing_config = graph.update_state(existing_config, {"messages": mock_tool_call_msg, "balance": mock_response['balance']}, as_node=tool_node_name)
    return graph.stream(None, config=existing_config, stream_mode="values")


def extract_node_name(graph, config):
    next_state = graph.get_state(config).next
    return next_state[0] if len(next_state) else "END"

def extract_recent_msg_info_and_tool_info(curr_node):
    last_message_in_curr_node = curr_node['messages'][-1]
    tool_calls_info = getattr(last_message_in_curr_node, 'tool_calls', None)
    tool_call_names = [tool_call['name'] for tool_call in tool_calls_info] if tool_calls_info else 'No Tool Call'
    print("Latest Message in state obj: ",last_message_in_curr_node.content, "| By: ",last_message_in_curr_node.__class__.__name__, " | Tool Call: ", tool_call_names)
    return last_message_in_curr_node.content, tool_calls_info, tool_call_names

@pytest.mark.asyncio
@pytest.mark.parametrize("msg_mocker", get_valid_messages())
async def test_valid_response(msg_mocker):
    graph = initialize_graph()
    print("Initialized Graph..")
    node_vs_tools = get_node_vs_tools_mapping()
    config = {"configurable": {"thread_id": str(random.randint(1, 188787))}}
    execution_path = graph.stream({"messages": msg_mocker['input_message'],
                                    "balance": msg_mocker['initial_balance']},
                                config=config,
                                stream_mode='values')
    print(f"Running the graph for the inp: {msg_mocker['input_message']}")
    curr_node = next(execution_path)
    nodes_visited = []
    tools_invoked = []
    final_msg = ""
    while curr_node:
        #node extraction
        node_name = extract_node_name(graph, config)
        nodes_visited.append(node_name)
        #tool extraction
        final_msg, tool_calls_info, tool_call_names = extract_recent_msg_info_and_tool_info(curr_node)
        tools_invoked.append(tool_call_names)
        #mock tool responses
        if tool_calls_info and len(tool_calls_info):
            #modify execution path to add the mocked tool response
            execution_path = update_path_with_mock_tool_call_response(graph, config, tool_calls_info, node_vs_tools, msg_mocker)
        #head to next node
        try:
            curr_node = next(execution_path)
        except StopIteration:
            print("Stream ended.")
            break
        except Exception as e:
            print(f"Error encountered: {e}")
            break
    print("Node Visited:  ", nodes_visited)
    print("Tool Invoked:  ", tools_invoked)
    print("Final Message: ", final_msg)
    #TODO add more sophisticated asserts
    assert "Payment processing is currently unavailable due to maintenance" in final_msg
