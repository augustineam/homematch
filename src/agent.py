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

from langchain.callbacks.tracers import ConsoleCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from pydantic import BaseModel, Field
from typing import List, Literal, Union, Any

from .utils import get_vector_store, get_retriever_tool

__hmgraph: Union[None, CompiledStateGraph] = None


class AgentState(MessagesState):
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

    user_preferences = {
        "budget": None,
        "bedrooms": None,
        "bathrooms": None,
        "amenities": [],
        "neighborhood": [],
        "style": None,
        "square_feet": None,
        "additional_notes": None,
    }

    def get_user_preferences_str() -> str:
        """Return a string representation of the user preferences."""

        features = []
        for key, value in user_preferences.items():
            if value:
                features.append(f"{key}: {value}")
            else:
                features.append(f"{key}: Not specified")

        if not features:
            return "No preferences have been set."

        return "\n".join(features)

    agent_tools = []
    embeddings = OpenAIEmbeddings()
    property_listings = get_vector_store(
        vecstore_path, properties_csv, "property_listings", embeddings, "row"
    )

    # Append retriever tool
    agent_tools.append(get_retriever_tool(property_listings, with_images))

    @tool
    def add_preference(key: str, value: Any) -> None:
        """Add a value to the preference dictionary. Creates the key if it doesn't exists."""
        key = key.lower()
        value = value.lower() if isinstance(value, str) else value
        old_value = user_preferences.get(key)

        # if key is not in user_preferences, add it
        if not old_value:
            user_preferences[key] = value
            return

        # If the new value is a list and the old value is not, convert the old value to a list
        if isinstance(value, list) and not isinstance(old_value, list):
            user_preferences[key] = [old_value] + value
            return

        # If the new value is not a list and the old value is, append the new value to the old value
        if not isinstance(value, list) and isinstance(old_value, list):
            if value not in old_value:
                old_value.append(value)
            user_preferences[key] = old_value
            return

        # If both are lists, merge them
        if isinstance(value, list) and isinstance(old_value, list):
            for v in value:
                if v not in old_value:
                    old_value.append(v)
            user_preferences[key] = old_value
            return

        # If both are not lists, replace the old value
        user_preferences[key] = value

    @tool
    def remove_preference(key: str, value: Any) -> None:
        """Remove a value from the preference dictionary."""
        key = key.lower()
        value = value.lower() if isinstance(value, str) else value

        # if key not in user_preferences, return
        if key not in user_preferences:
            return

        # if value is not in user_preferences[key], return
        if isinstance(user_preferences[key], list):
            if value in user_preferences[key]:
                user_preferences[key].remove(value)
        else:
            user_preferences[key] = None

    @tool
    def update_preference(key: str, value: Any) -> None:
        """Update a value in the preference dictionary."""
        if key in user_preferences:
            user_preferences[key] = value

    # Append user preference tools
    agent_tools.append(add_preference)
    agent_tools.append(remove_preference)
    agent_tools.append(update_preference)

    async def agent(state: AgentState):
        """Call the model's agent with the current state"""

        model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.5, max_tokens=1000
        ).bind_tools(agent_tools)

        preferences = get_user_preferences_str()
        summary = state.get("summary", "No conversation summary provided.")

        if with_images:
            image_placeholder = """**Row number placeholder:**
Each property listing will be presented with a row number.
Add a row number placeholder at the end of each listing.

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
"""
        else:
            image_placeholder = ""

        system_prompt = f"""
You are a personalized real estate agent with a passion for helping users find their ideal home. Be charismatic and friendly. Make the user feel comfortable and engaged, and gather as much information from the user as possible to provide the best possible recommendations.

**Your main goals are:**
1. To know and update the user preferences based on your interactions.
2. To offer the user relevant property listings based on their preferences.
3. Present the most relevant property listings to the user when:
   - You think you have enough information.
   - The user preferences have been updated.

**Structured Questions to Ask:**
1. What is your budget for the property?
2. How many bedrooms and bathrooms are you looking for?
3. Do you have a preferred style or type of home?
4. Are there any specific amenities you require, such as a backyard or garage?
5. Do you prefer a specific neighborhood or proximity to certain locations like schools or parks?

**User Preferences Rules:**
- If no house preference has been provided, start by asking questions to the user.
- The preferences must describe the property, the price, budget, or neighborhood description, property style, etc.
- You will be able to add, update and remove preferences from the preference dictionary.
- It is **strictly prohibited** to save information unrelated to the property or the neighborhood.
- It is **strictly prohibited** to add preferences already mentioned in the preference dictionary.

**Preference Management Guidelines:**
- Store preferences in a structured format (e.g., dictionary) to categorize different aspects (budget, bedrooms, bathrooms, amenities, neighboorhood, style, etc.).
- Before adding a new preference, check if it already exists in the preferences dictionary.
- If a preference is updated, replace the old value with the new one.
- Remove preferences if the user changes their mind or no longer requires them.

**Retrieving Property Listings:**
- Once you have gathered enough information about the user preferences, retrieve relevant property listings that match their criteria.
- Present these listings to the user in a clear and engaging manner.
- Ensure to highlight key features of each property that align with the user preferences.

{image_placeholder}
**User Preferences:**
{preferences}

**Conversation Summary:**
{summary}
"""
        system_message = SystemMessage(content=system_prompt)
        # If summary exists, there are two messages left from previous summarization
        # remove them from the conversation, otherwise use the whole conversationstem_prompt)
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
You are an AI assistant for a real state AI agent, and your task is to help the agent remember important information about the conversation with its clients.
The agent is having a conversation with a user looking to buy a property.
Distill the main interaction ideas to help the AI real estate agent to understand the user preferences and needs.
Make sure to capture important information about the user's ideal home, their budget, and any specific requirements they have mentioned.

You will produce one or many ideas formated as a JSON object `{{ "summary": ["sentence1", "sentence2", ...] }}`.

**Format Instructions:**
{format_instructions}

Strip the 3 backticks from the response.
"""
        existing_summary = state.get("summary", "")
        system_message = SystemMessage(content=system_prompt)

        try:
            # Leave 2 messages in the conversation history to avoid context window overflow.
            response = model.invoke([system_message] + state["messages"][:-2])
            response: SummaryResponse = parser.invoke(response)
        except:
            # If the model fails to parse the response
            return dict(summary=existing_summary)

        new_summary = [existing_summary] if existing_summary else []
        new_summary.extend(response.summary)

        # Keep last two messages in the conversation history
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


async def cli_achat(vecstore_path: str, csv_path: str, trace: bool):
    """Run the agent from the command line."""

    callbacks = [ConsoleCallbackHandler()] if trace else []

    agent = HomeMatchAgent(
        vecstore_path,
        csv_path,
        config=dict(configurable=dict(thread_id=4), callbacks=callbacks),
    )

    print(agent.get_graph().draw_ascii())
    input_message = SystemMessage(content="The user has started the conversation.")

    while True:
        print("Agent: ", end="", flush=True)
        async for msg, metadata in agent.astream(
            dict(messages=[input_message]), stream_mode="messages"
        ):
            if (
                isinstance(msg, AIMessageChunk)
                and metadata["langgraph_node"] == "agent"
            ):
                finish_reason = msg.response_metadata.get("finish_reason", "")
                print() if finish_reason else print(msg.content, end="", flush=True)

        print()
        user_input = ""
        while not user_input:
            user_input = input("User: ")

        if user_input == "/exit":
            break

        input_message = HumanMessage(content=user_input)
