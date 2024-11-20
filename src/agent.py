from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    RemoveMessage,
    AIMessage,
    ToolMessage,
    HumanMessage,
    AIMessageChunk,
)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, InjectedState
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from pydantic import BaseModel, Field
from typing import Annotated, Dict, List, Literal, Union

from .utils import get_vector_store, get_retriever_tool

import json


__hmgraph: Union[None, CompiledStateGraph] = None


class AgentState(MessagesState):
    preferences: str = ""
    summary: str = ""


def HomeMatchAgent(
    vecstore_path: str,
    properties_csv: str,
    with_images: bool = False,
    agent_executor: bool = False,
    config: Union[RunnableConfig, None] = None,
) -> CompiledStateGraph:
    """Create the HomeMatch Real State AI Agent LangGraph.

    Args:
        vecstore_path (str): Path to the vector store.
        properties_csv (str): Path to the properties csv.
        imgprompts_csv (str | None): Path to the imgprompts csv.

    Returns:
        CompiledStateGraph: The compiled state graph.
    """

    global __hmgraph

    if __hmgraph:
        return __hmgraph

    agent_tools = []
    embeddings = OpenAIEmbeddings()
    property_listings = get_vector_store(
        vecstore_path, properties_csv, "property_listings", embeddings, "row"
    )

    # Append retriever tool
    agent_tools.append(get_retriever_tool(property_listings, with_images))

    @tool
    def update_user_preferences(
        preferences: str, state: Annotated[Dict, InjectedState]
    ):
        """Update the user preferences in the state"""

        state.update(dict(preferences=preferences))
        return "User preferences updated successfully."

    # Append user preference update tool
    agent_tools.append(update_user_preferences)

    async def agent(state: AgentState):
        """Call the model's agent with the current state"""

        model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.5, max_tokens=1000
        ).bind_tools(agent_tools)

        house_preferences = state.get("preferences", "No preferences provided.")
        summary = state.get("summary", "No conversation summary provided.")

        system_prompt = f"""
You are a real estate agent with a passion for helping users find their ideal home. Be charismatic, friendly, and optimistic.

Probably, some details about the user won't be necessary, but it will be a nice touch to really know the user. For example, ask what their name is, if they have any kids or pets, and what their favorite color is. You might find this information useful to make great suggestions.

The System will keep a record of the user preferences and make it available to you in the **House Preferences** section below.
The System will keep a record of the conversation history available to you in the **Conversation Summary** section below.

**Your main goals are:**
- To know and update the user preferences based on your interactions.
- To offer the user relevant property listings based on their preferences.
- Present the most relevant property listings to the user when:
- You think you have enough information.
- The user preferences have been updated.

**House Preferences Rules:**
- It is strictly prohibited to save information unrelated to the property or the neighborhood.
- The preferences must describe the property, the price, budget, or neighborhood description.
- If no house preference has been provided, start by asking questions to the user.
- Update the preferences if the user shares new information that describes the property, the price, or neighborhood. Ensure to add new information to existing preferences rather than overriding them.
- Always confirm with the user before updating the preferences.

**Structured Questions to Ask:**
1. What is your budget for the property?
2. How many bedrooms and bathrooms are you looking for?
3. Do you have a preferred style or type of home?
4. Are there any specific amenities you require, such as a backyard or garage?
5. Do you prefer a specific neighborhood or proximity to certain locations like schools or parks?

**Retrieving Property Listings:**
- Once you have gathered enough information about the user preferences, retrieve relevant property listings that match their criteria.
- Present these listings to the user in a clear and engaging manner.
- Ensure to highlight key features of each property that align with the user preferences.

**IMPORTANT: Listings with Row:**
When the listing is retrieved with row number, follow this instruction:

- After presenting each listing, append the text `<<row_number>>` below the listing. This helps the system place the listing images correctly.

For example:

```plaintext
1. **Colonial Style Home in the Suburbs**
   - Price: $800,000
   - Bedrooms: 4
   - Bathrooms: 3
   - Sqft: 3000
   - Description: Beautiful colonial style home with traditional charm and modern updates. Large backyard perfect for outdoor entertaining.
   - Neighborhood: Nestled in a quiet suburban area, close to schools and parks.
<<34>>
```

**House Preferences:**
{house_preferences}

**Conversation Summary:**
{summary}
"""
        system_message = SystemMessage(content=system_prompt)
        response = await model.ainvoke([system_message] + state["messages"])

        # Remove tool messages and tool calls from the chat history
        remove_messages = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
        ]

        # Return the state update
        return dict(messages=remove_messages + [response])

    def should_continue(
        state: AgentState,
    ) -> Literal["tools", "summarize", "__end__"]:
        """Condition to decide whether summarize, call a tool or end AI turn."""

        messages = state.get("messages")
        # Go to the tools node if the last message is a tool call
        if messages[-1].tool_calls:
            return "tools"

        # Go to summarize node if there are more than 6 messages in the history
        if 6 < len(messages):
            return "summarize"

        return END

    def distill_interactions(state: AgentState):
        """Distill the interaction between the agent and the user."""

        class SummaryResponse(BaseModel):
            summary: List[str] = Field(
                description="list of distilled interaction key points"
            )

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=512)
        parser = PydanticOutputParser(pydantic_object=SummaryResponse)
        format_instructions = parser.get_format_instructions()

        system_prompt = f"""
This is a friendly conversation between an AI real estate agent and a user looking to buy a home.

Your task is to distill the main ideas of the interaction, keeping all relevant details about the conversation. Ensure that you capture any new preferences mentioned by the user, even if they have not been saved yet.

You will produce one or more ideas and follow the format instructions to the letter.

**Format Instructions:**
{format_instructions}
"""
        system_message = SystemMessage(content=system_prompt)

        response = model.invoke([system_message] + state["messages"])
        response: SummaryResponse = parser.invoke(response)

        existing_summary = state.get("summary", "")

        new_summary = [existing_summary] if existing_summary else []
        new_summary.extend(response.summary)

        # keep last two message and remove the rest
        remove_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

        # Return the state updates.
        return dict(summary="\n".join(new_summary), messages=remove_messages)

    # >>> print(agent.get_graph().draw_ascii())
    #                   +-----------+
    #                   | __start__ |
    #                   +-----------+
    #                          *
    #                          *
    #                          *
    #                     +-------+
    #                     | agent |.
    #                  ...+-------+ ....
    #               ...        .        ....
    #           ....           .            ...
    #         ..               .               ....
    # +-------+         +-----------+              ..
    # | tools |         | summarize |             ..
    # +-------+         +-----------+           ..
    #                              **         ..
    #                                **     ..
    #                                  *   .
    #                               +---------+
    #                               | __end__ |
    #                               +---------+

    memory = MemorySaver()
    workflow = StateGraph(AgentState)

    tool_node = ToolNode(agent_tools)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("summarize", distill_interactions)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("summarize", END)

    hmagent = workflow.compile(checkpointer=memory)

    if config:
        hmagent = hmagent.with_config(config)

    if agent_executor:
        from langchain.agents import AgentExecutor

        __hmgraph = AgentExecutor(agent=hmagent, tools=agent_tools)
    else:
        __hmgraph = hmagent

    return __hmgraph


async def cli_achat(vecstore_path: str, csv_path: str):
    """Run the agent from the command line."""

    agent = HomeMatchAgent(vecstore_path, csv_path)
    print(agent.get_graph().draw_ascii())

    config = dict(configurable=dict(thread_id=4))
    input_message = SystemMessage(content="The user has started the conversation.")

    while True:
        print("Agent: ", end="", flush=True)
        async for msg, _ in agent.astream(
            dict(messages=[input_message]), config, stream_mode="messages"
        ):
            if isinstance(msg, AIMessageChunk):

                # FIXME: It is possible we are getting the a chunk with summary json...
                # But what if the Agent sends a similar content that is not the summary?
                # In whispers, he falls
                # Empty nest, broken promise
                # Honor's echo flees.
                if msg.content.startswith("{") or json_chunk:
                    json_chunk = json_chunk + msg.content
                    if msg.content.endswith("}"):
                        try:
                            summary: Dict = json.loads(json_chunk)
                            if "summary" not in summary:
                                gathered = gathered + json_chunk
                            else:
                                print("Summary:\n", "\n".join(summary["summary"]))
                                # FIXME: For some reason the summary is not updated
                                # after adding the json_chunk fix...
                                agent.update_state(config, values=summary)
                        except json.decoder.JSONDecodeError:
                            continue
                    else:
                        continue
                else:
                    finish_reason = msg.response_metadata.get("finish_reason", "")
                    print() if finish_reason else print(msg.content, end="", flush=True)

        print()
        user_input = ""
        while not user_input:
            user_input = input("User: ")

        if user_input == "/exit":
            break

        input_message = HumanMessage(content=user_input)
