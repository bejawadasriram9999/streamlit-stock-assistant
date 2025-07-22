import streamlit as st
import json
import requests # For making HTTP requests to the Gemini API
import os # For accessing environment variables
from dotenv import load_dotenv # For loading environment variables from .env file
import yfinance as yf # For fetching stock data
import asyncio # For running async functions

# Load environment variables from .env file
load_dotenv()

# Define the agent's properties (simulated)
AGENT_NAME = "Stock_Market_Assistant"
AGENT_MODEL = "gemini-2.0-flash" # Using gemini-2.0-flash directly for simulation
AGENT_DESCRIPTION = "A helpful stock market assistant that can fetch real-time stock prices."
AGENT_INSTRUCTION = """You are a helpful stock market assistant.
You have access to a tool called `get_stock_info` that can fetch real-time stock data.
Use this tool when the user asks for current stock prices or information about a specific stock ticker.
When using the tool, provide the ticker symbol (e.g., 'AAPL', 'GOOGL').
If you don't know something, just say so.
Be concise and answer directly based on the information provided by the tool or your knowledge.
"""

# --- Streamlit App UI ---
st.set_page_config(page_title=f"📈 {AGENT_NAME.replace('_', ' ').title()}", layout="centered")

st.title(f"📈 {AGENT_NAME.replace('_', ' ').title()}")
st.markdown(f"*{AGENT_DESCRIPTION}*")
st.markdown("---")

# Retrieve API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# Display a warning if API key is not found
if not api_key:
    st.warning("Gemini API Key not found. Please create a .env file with GOOGLE_API_KEY=your_api_key, or set it as an environment variable.")
    st.markdown("Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Tool Function: Fetch Stock Info ---
def get_stock_info(ticker: str) -> str:
    """
    Fetches real-time stock information for a given ticker symbol.
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL').
    Returns:
        str: A JSON string containing stock information or an error message.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return json.dumps({"error": f"Could not find information for ticker: {ticker}. Please check the symbol."})

        # Extract relevant real-time data
        # Note: 'currentPrice' or 'regularMarketPrice' might not always be available instantly.
        # 'previousClose' is often more reliable for recent data.
        data = {
            "ticker": ticker.upper(),
            "longName": info.get("longName", "N/A"),
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "currentPrice": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "previousClose": info.get("previousClose", "N/A"),
            "open": info.get("open", "N/A"),
            "dayHigh": info.get("dayHigh", "N/A"),
            "dayLow": info.get("dayLow", "N/A"),
            "volume": info.get("volume", "N/A"),
            "marketCap": info.get("marketCap", "N/A")
        }
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"error": f"An error occurred while fetching data for {ticker}: {str(e)}"})

# --- Gemini API Tool Schema Definition ---
# This describes the `get_stock_info` function to the LLM
STOCK_INFO_TOOL_SCHEMA = {
    "name": "get_stock_info",
    "description": "Fetches real-time stock information for a given ticker symbol. Returns current price, previous close, open, high, low, and volume.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "ticker": {
                "type": "STRING",
                "description": "The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')."
            }
        },
        "required": ["ticker"]
    }
}

# --- Function to interact with the simulated agent (Gemini API) ---
async def get_agent_response(prompt: str, api_key: str):
    """
    Simulates the agent's response by calling the Gemini API with agent's instructions and tools.
    Handles tool calls and returns the final response.
    """
    if not api_key:
        return "Gemini API Key is missing. Please provide it to use the assistant."

    # Initial prompt to the LLM with agent instruction and user query
    # The agent instruction is sent as a user part to guide the model's behavior.
    chat_history = [
        {"role": "user", "parts": [{"text": AGENT_INSTRUCTION}]},
        {"role": "user", "parts": [{"text": prompt}]}
    ]

    payload = {
        "contents": chat_history,
        "tools": [
            {"functionDeclarations": [STOCK_INFO_TOOL_SCHEMA]}
        ]
    }
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{AGENT_MODEL}:generateContent?key={api_key}"

    try:
        # First call to LLM: User prompt + Tool definition
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        # Check if the LLM decided to call a tool
        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("functionCall"):

            function_call = result["candidates"][0]["content"]["parts"][0]["functionCall"]
            function_name = function_call["name"]
            function_args = function_call["args"]

            if function_name == "get_stock_info":
                st.info(f"Agent wants to fetch stock info for: {function_args.get('ticker')}")
                # Execute the local Python function
                tool_output = get_stock_info(function_args.get("ticker"))

                # Append the LLM's function call to history
                # This is crucial for the LLM to understand its own previous turn
                chat_history.append({
                    "role": "model", # The model's turn
                    "parts": [{"functionCall": function_call}] # The actual function call object
                })

                # Send the tool output back to the LLM for a final response
                chat_history.append({
                    "role": "function",
                    "parts": [{"functionResponse": {"name": "get_stock_info", "response": json.loads(tool_output)}}]
                })

                # Second call to LLM: Original prompt + Model's function call + Tool output
                payload_with_tool_output = {
                    "contents": chat_history,
                    "tools": [
                        {"functionDeclarations": [STOCK_INFO_TOOL_SCHEMA]}
                    ]
                }
                response_final = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload_with_tool_output))
                response_final.raise_for_status()
                result_final = response_final.json()

                if result_final.get("candidates") and result_final["candidates"][0].get("content") and \
                   result_final["candidates"][0]["content"].get("parts") and \
                   result_final["candidates"][0]["content"]["parts"][0].get("text"):
                    return result_final["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return "I received data from the stock tool, but couldn't formulate a clear response."
            else:
                return f"Agent tried to call an unknown function: {function_name}"
        else:
            # If no tool call, return the direct text response from the first LLM call
            if result.get("candidates") and result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               result["candidates"][0]["content"]["parts"][0].get("text"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "I couldn't get a clear response from the model. Please try again."

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the Gemini API: {e}")
        return f"I'm currently unable to provide a response due to a technical issue: {e}. Please ensure your API key is correct and has access to the '{AGENT_MODEL}' model."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."


# Accept user input
if prompt := st.chat_input("Ask me about the stock market... (e.g., 'What is the price of AAPL?')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the async function synchronously for Streamlit's context
            response = asyncio.run(get_agent_response(prompt, api_key))
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
