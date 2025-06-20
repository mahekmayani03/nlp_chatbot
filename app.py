import streamlit as st
import pickle
import json
import random

# Configure the page
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

    
# Load the trained model and vectorizer
#@st.cache_resource
def load_model_and_data():
    # Load the trained model and vectorizer
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Load the intents data
    with open('dataset/intents.json', 'r') as f:
        intents = json.load(f)

    return model, vectorizer, intents

# Initialize the model and data
try:
    best_model, vectorizer, intents = load_model_and_data()
    model_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.error("Please make sure the model files and intents.json are in the correct directories.")
    model_loaded = False


def chatbot_response(user_input):
    """Generate chatbot response based on user input"""
    if not model_loaded:
        return "Sorry, the chatbot model is not loaded properly."

    try:
        # Transform the input text
        input_text = vectorizer.transform([user_input])
    
        # Predict the intent
        predicted_intent = best_model.predict(input_text)[0]
    
        # Find the matching intent and get a random response
        for intent in intents['intents']:
            if intent['tag'] == predicted_intent:
                response = random.choice(intent['responses'])
                return response

        # Default response if no intent matches
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
def main():
    st.title("🤖 Chatbot")
    st.markdown("Welcome! Ask me anything and I'll try to help you.")


    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        if model_loaded:
            response = chatbot_response(prompt)
        else:
            response = "Sorry, the chatbot is not available right now."

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Sidebar with additional options
    with st.sidebar:
        st.header("Options")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    

if __name__ == "__main__":
    main()
