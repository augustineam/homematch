from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    RemoveMessage,
    AIMessage,
    HumanMessage,
    AIMessageChunk,
)

from langchain.callbacks.tracers import ConsoleCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from pydantic import BaseModel, Field
from typing import List, Literal, Union

from .utils import (
    get_vector_store,
    get_retriever_tool,
    get_image_vector_store,
    get_retriever_by_image,
)

__hmgraph: Union[None, CompiledStateGraph] = None


class AgentState(MessagesState):
    summary: str = ""
    preferences: str = ""


def isHumanOrAIMessage(message: BaseMessage) -> bool:
    """Check if a message is from a human or an AI but not a tool call."""
    return isinstance(message, HumanMessage) or (
        isinstance(message, AIMessage) and not message.tool_calls
    )


def HomeMatchAgent(
    vecstore_path: str,
    properties_csv: str,
    with_images: bool = False,
    agent_executor: bool = False,
    config: Union[RunnableConfig, None] = None,
    image_index_data: str | None = None,
    image_index_method: Literal["images", "descriptions"] = "descriptions",
) -> CompiledStateGraph:
    """Create the HomeMatch Real State AI Agent LangGraph.

    Args:
        vecstore_path (str): Path to the vector store.
        properties_csv (str): Path to the properties csv.
        imgprompts_csv (str | None): Path to the imgprompts csv.
        with_images (bool): Whether to add image placeholder instruction.
        agent_executor (bool): Whether to use the agent executor.
        config (RunnableConfig | None): The runnable configuration.
        image_index_data(str | None): Path to the data used to index the images.
        image_index_method (Literal["images", "descriptions"]): The method to use for indexing images.

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

    if with_images and image_index_data:
        image_vector_store = get_image_vector_store(
            vecstore_path, image_index_data, image_index_method
        )
        agent_tools.append(
            get_retriever_by_image(image_vector_store, property_listings)
        )

    async def agent(state: AgentState):
        """Call the model's agent with the current state"""

        model = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.5, max_tokens=1000
        ).bind_tools(agent_tools)

        if with_images:
            image_placeholder = """- If the user shares an image, use the tool to find a similar property based on the image.
- Make sure to include the row number placeholder at the end of each listing, **this is very important**.

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
You are a personalized real estate AI agent with a passion for helping users find their ideal home. Be charismatic and friendly.
Make the user feel comfortable and engaged, and gather as much information from the user as possible to provide the best possible recommendations.
The system collects and keeps the user preferences in the **User Preferences** section.
The system summarizes and keeps your conversation with the user in the **Conversation Summary** section.

**Your main goals are:**

1. Ask the user what are their preferences.
2. Retrieve property listings that match the user preferences.
3. Show the most relevant property listings to the user that match their criteria.
4. Present the listings to the user immediately as soon as you have the needed information.

To complete the task successfully you have to use the information provided in the summary, the preferences and the chat history to retrieve and provide the best recommendations possible to the user.

**Structured Questions to Ask:**

1. What is your budget for the property?
2. How many bedrooms and bathrooms are you looking for?
3. Do you have a preferred style or type of home?
4. Are there any specific amenities you require, such as a backyard or garage?
5. Do you prefer a specific neighborhood or proximity to certain locations like schools or parks?

**Retrieving Property Listings:**

- Retrieve relevant property listings that match their criteria.
- Present these listings to the user in a clear and engaging manner.
- Ensure to highlight key features of each property that align with the user preferences.
{image_placeholder}
"""
        system_message = SystemMessage(content=system_prompt)
        # If summary exists, there are two messages left from previous summarization
        # remove them from the conversation, otherwise use the whole conversationstem_prompt)
        response = await model.ainvoke([system_message] + state["messages"])

        # Return the state update
        return dict(messages=[response])

    def should_continue(
        state: AgentState,
    ) -> Literal["tools", "assistant", "__end__"]:
        """Condition to decide whether summarize, call a tool or end AI turn."""

        messages = state.get("messages")

        # Go to the tools node if the last message is a tool call
        if messages[-1].tool_calls:
            return "tools"

        # The assistant will summarize the last 10 messages in the conversation
        # If summary exists, there are two messages left from previous summarization
        summary = state.get("summary", "")
        msg_threshold = 12 if summary else 10

        # Get number of AI (without tools) & Human messages.
        n_messages = len([m for m in messages if isHumanOrAIMessage(m)])

        # Go to summarize node if there are more than 6 messages in the history
        if msg_threshold < n_messages:
            return "assistant"

        return END

    def assistant(state: AgentState):
        """Call the summary and preference assistant with the current state and return the summary and preferences."""

        class AssistantResponse(BaseModel):
            summary: List[str] = Field(description="list of summary sentences")
            preferences: str = Field(description="current user preferences")

        summary = state.get("summary", "")
        preferences = state.get("preferences", "")

        model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.1, max_tokens=512)
        parser = PydanticOutputParser(pydantic_object=AssistantResponse)
        format_instructions = parser.get_format_instructions()

        system_prompt = f"""
[END OF CONVERSATION]

---
**Previous Preferences:**
{preferences}
---

You are an AI assistant for a real estate AI agent, tasked with helping the agent remember important information from client conversations.
The messages above are from a conversation between the AI agent and a user looking to buy a property.
Your goal is to create a concise summary of the conversation and capture the user's current preferences in a short sentence.

For the summary:
- Produce a list of concise summary sentences based on the conversation provided above.
- If the user shows interest in any listings mentioned by the agent, include a brief summary of those listings.
- Prioritize the most recent and relevant information.
- If the conversation lacks sufficient information, indicate that more details are needed.

For the preferences:
- Generate a concise sentence capturing the user's current expectations for their ideal home.
- Focus on key aspects like budget, location, size, and specific requirements.
- Prioritize the most recently expressed preferences, in case of contradictions or changes.
- If the conversation lacks clear preferences, provide a default response indicating that more information is required.

To assist you, the preferences from last time is provided as context above.

Please use the following format for your response:
{format_instructions}

Strip the 3 backticks from the response.
"""

        try:
            # If we have a summary from last time, then skip the first two messages from the conversation
            messages = [m for m in state["messages"] if isHumanOrAIMessage(m)]
            messages = messages[2:] if summary else messages

            history = ["[START OF CONVERSATION]"] + [
                f"{m.type}: {m.content}" for m in messages
            ]

            response = model.invoke("\n".join(history + [system_prompt]))
            response: AssistantResponse = parser.invoke(response)
        except Exception as e:
            print(e)
            # If the model fails to parse the response, update nothing
            return dict(messages=[])

        new_summary = [summary] if summary else []
        new_summary.extend(response.summary)

        summary = "\n".join(new_summary)
        preferences = response.preferences

        assistent_message = f"""
**Preferences so far:**
{preferences}

**Summary so far:**
{summary}
"""

        # AI (without tool calls) & Human messages
        messages = [m for m in state["messages"] if isHumanOrAIMessage(m)]

        # Assistant message
        system_message = [SystemMessage(content=assistent_message)]

        # Remove all messages except for the last two messages
        keep = [m.id for m in messages[-2:]]
        remove_messages = [
            RemoveMessage(id=m.id) for m in state["messages"] if m.id not in keep
        ]

        # The last message contains the assistant summary and the current user preferences
        messages = remove_messages + system_message

        # Return the state updates.
        return dict(messages=messages, preferences=preferences, summary=summary)

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
    # | tools |         | assistant |             ..
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
    workflow.add_node("assistant", assistant)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("assistant", END)

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
