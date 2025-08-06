import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import os

st.set_page_config(page_title="LangChain ChatBot with GroqAI", page_icon="🤖")
st.title("Simple LangChain ChatBot with GroqAI 🤖")

with st.sidebar:
    st.header("Configuration")
    ## APi Key
    api_key=st.text_input("GROQ API Key", type="password",help="GET Free API Key at console.groq.com")
    
    # model selection
    model = st.selectbox("Select Model",
                          ["llama3-8b-8192", "gemma2-9b-it"],
                          index = 0
    )
    # clear chat 
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_chain(api_key,model):
    if not api_key:
        return None
    
    # Initialize the Groq LLM with the provided API key and model
    # Set stream=True to enable streaming responses
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model,
        temperature=0.5,
        stream=True,)
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ])

    # Create the chain with the prompt and output parser
    chain = prompt | llm | StrOutputParser()
    return chain

# get the chain
chain = get_chain(api_key, model)
if not chain:
    st.error("Please enter your Groq API key in the side bar to initialize the model.")
    st.markdown("[Get your free API key here](https://console.groq.com)")

# Display chat messages
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

        #chain input
    if prompt := st.chat_input("Ask a question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        try :
            # Get the response from the chain
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in chain.stream({"input": prompt}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")  # Show the response as it streams
                message_placeholder.markdown(full_response)

                # Append the response to the session state
                st.session_state.messages.append({"role": "assistant", "content": full_response})


        except Exception as e:
            st.error(f"Error: {str(e)}")

# Example Questions
st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("What is LangChain?")
    st.markdown("How does GroqAI work?")
with col2:
    st.markdown("What is the capital of France?")
    st.markdown("Tell me a joke.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Umar Faizan")
st.markdown("Connect with me on [LinkedIn](https://www.linkedin.com/in/u-faizan)")
st.markdown("Check out the code on [GitHub]")
st.markdown("Powered by [LangChain](https://langchain.com) and [GroqAI](https://groq.com)")



    


