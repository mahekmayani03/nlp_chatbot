import streamlit as st
import pickle
import json
import random
import speech_recognition as sr
import pyttsx3
import tempfile
import os
import base64

# Configure the page
st.set_page_config(
    page_title="Speech-to-Speech Chatbot",
    page_icon="🤖🎤",
    layout="centered"
)

# Initialize TTS engine
@st.cache_resource
def init_tts_engine():
    """Initialize text-to-speech engine"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        return engine
    except Exception as e:
        st.error(f"TTS initialization error: {e}")
        return None

# Load the trained model and vectorizer
@st.cache_resource
def load_model_and_data():
    try:
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('dataset/intents.json', 'r') as f:
            intents = json.load(f)
        return model, vectorizer, intents
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

# Initialize components
best_model, vectorizer, intents = load_model_and_data()
model_loaded = all([best_model, vectorizer, intents])
tts_engine = init_tts_engine()

def speech_to_text():
    """Convert speech to text using microphone"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
        
        st.info("🔄 Processing speech...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        return "Timeout: No speech detected"
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

def text_to_speech(text):
    """Convert text to speech"""
    if tts_engine is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            temp_filename = tmp_file.name
        tts_engine.save_to_file(text, temp_filename)
        tts_engine.runAndWait()
        with open(temp_filename, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        os.unlink(temp_filename)
        return audio_bytes
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def play_audio_autoplay(audio_bytes):
    """Create HTML audio element with autoplay"""
    if audio_bytes:
        try:
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio autoplay style="display:none;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Autoplay error: {str(e)}")

def chatbot_response(user_input):
    """Generate chatbot response based on user input"""
    if not model_loaded:
        return "Sorry, the chatbot model is not loaded properly."
    try:
        input_text = vectorizer.transform([user_input])
        predicted_intent = best_model.predict(input_text)[0]
        for intent in intents['intents']:
            if intent['tag'] == predicted_intent:
                return random.choice(intent['responses'])
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("🤖🎤 Speech-to-Speech Chatbot")
    st.markdown("Welcome! Choose your preferred input method and start chatting.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "speech_input" not in st.session_state:
        st.session_state.speech_input = ""
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "audio" in message:
                st.audio(message["audio"], format='audio/wav')
    
    # Custom styling
    st.markdown("""
    <style>
    .input-mode-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border: 2px solid #e9ecef;
    }
    .mode-title {
        font-size: 18px;
        font-weight: 600;
        color: #333;
        margin-bottom: 15px;
        text-align: center;
    }
    .combined-input-box {
        display: flex;
        align-items: center;
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 25px;
        padding: 12px 60px 12px 20px;
        margin: 15px 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .combined-input-box:hover {
        border-color: #007bff;
        box-shadow: 0 4px 12px rgba(0,123,255,0.15);
    }
    .combined-text-input {
        flex: 1;
        border: none;
        outline: none;
        font-size: 16px;
        padding: 8px 0;
        font-family: inherit;
        background: transparent;
    }
    .combined-mic-btn {
        position: absolute;
        right: 15px;
        width: 40px;
        height: 40px;
        border: none;
        border-radius: 50%;
        background: #007bff;
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        font-size: 16px;
    }
    .combined-mic-btn:hover {
        background: #0056b3;
        transform: scale(1.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Input method selector
    st.markdown("### 🎯 Choose Your Input Method")
    input_method = st.radio(
        "",
        options=["💬 Text Only", "🎤 Speech Only", "🔄 Both (Text + Speech)"],
        horizontal=True,
        key="input_method"
    )
    
    prompt = None
    
    # TEXT ONLY MODE
    if input_method == "💬 Text Only":
        st.markdown('<div class="input-mode-container">', unsafe_allow_html=True)
        st.markdown('<div class="mode-title">💬 Text Input Mode</div>', unsafe_allow_html=True)
        st.markdown("Type your message and press Enter to send")
        prompt = st.chat_input("💬 Type your message here...")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SPEECH ONLY MODE
    elif input_method == "🎤 Speech Only":
        st.markdown('<div class="input-mode-container">', unsafe_allow_html=True)
        st.markdown('<div class="mode-title">🎤 Speech Input Mode</div>', unsafe_allow_html=True)
        st.markdown("Click the button below and speak your message")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎤 Start Speaking", use_container_width=True, type="primary"):
                if not model_loaded:
                    st.error("❌ Chatbot model not loaded!")
                else:
                    with st.spinner("🎤 Listening... Speak now!"):
                        speech_text = speech_to_text()
                        if speech_text and not any(error in speech_text for error in ["Timeout", "Could not", "Error"]):
                            st.session_state.speech_input = speech_text
                            st.success(f"✅ Recognized: {speech_text}")
                        else:
                            st.error(f"❌ {speech_text}")
        
        # Show recognized speech with action buttons
        if st.session_state.speech_input:
            st.info(f"🎤 Recognized: **{st.session_state.speech_input}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Send", key="send_speech", use_container_width=True, type="primary"):
                    prompt = st.session_state.speech_input
                    st.session_state.speech_input = ""
            with col2:
                if st.button("🔄 Re-record", key="rerecord", use_container_width=True):
                    st.session_state.speech_input = ""
                    st.rerun()
            with col3:
                if st.button("❌ Cancel", key="cancel_speech", use_container_width=True):
                    st.session_state.speech_input = ""
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # BOTH MODES COMBINED
    else:
        st.markdown('<div class="input-mode-container">', unsafe_allow_html=True)
        st.markdown('<div class="mode-title">🔄 Combined Input Mode</div>', unsafe_allow_html=True)
        st.markdown("Type your message OR click the microphone to speak")
        
        # Combined input interface
        col1, col2 = st.columns([6, 1])
        with col1:
            text_input = st.text_input("", placeholder="Type your message here...", key="combined_text", help="Type and press Enter")
            if text_input:
                prompt = text_input
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("🎤", key="combined_mic", help="Click to speak", use_container_width=True):
                if not model_loaded:
                    st.error("❌ Chatbot model not loaded!")
                else:
                    with st.spinner("🎤 Listening..."):
                        speech_text = speech_to_text()
                        if speech_text and not any(error in speech_text for error in ["Timeout", "Could not", "Error"]):
                            st.session_state.speech_input = speech_text
                            st.success(f"✅ Speech recognized: {speech_text}")
                            # Set the speech as prompt for immediate processing
                            prompt = speech_text
                        else:
                            st.error(f"❌ {speech_text}")
        
        # Show speech input if available
        if st.session_state.speech_input and not prompt:
            st.info(f"🎤 Speech ready: **{st.session_state.speech_input}**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Send Speech", key="send_speech_combined", use_container_width=True):
                    prompt = st.session_state.speech_input
                    st.session_state.speech_input = ""
            with col2:
                if st.button("❌ Clear Speech", key="clear_speech_combined", use_container_width=True):
                    st.session_state.speech_input = ""
                    st.rerun()
        
        # Fallback regular chat input
        if not prompt:
            fallback_prompt = st.chat_input("Or type here as backup...")
            if fallback_prompt:
                prompt = fallback_prompt
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process the message
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        if model_loaded:
            response = chatbot_response(prompt)
        else:
            response = "Sorry, the chatbot is not available right now."
        
        # Generate speech for response (only in speech modes)
        audio_bytes = None
        if input_method in ["🎤 Speech Only", "🔄 Both (Text + Speech)"] and tts_engine:
            with st.spinner("🔊 Generating speech..."):
                audio_bytes = text_to_speech(response)
        
        # Add assistant response to chat history
        message_data = {"role": "assistant", "content": response}
        if audio_bytes:
            message_data["audio"] = audio_bytes
        
        st.session_state.messages.append(message_data)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/wav')
                play_audio_autoplay(audio_bytes)
        
        # Clear any remaining speech input
        if "speech_input" in st.session_state:
            st.session_state.speech_input = ""
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.speech_input = ""
            st.rerun()
        
        st.markdown("---")
        
        # Current mode display
        st.subheader("🎯 Current Mode")
        mode_colors = {
            "💬 Text Only": "🟢", 
            "🎤 Speech Only": "🔵", 
            "🔄 Both (Text + Speech)": "🟡"
        }
        st.markdown(f"{mode_colors.get(input_method, '⚪')} **{input_method}**")
        
        st.markdown("---")
        
        # Speech settings (only for speech modes)
        if input_method in ["🎤 Speech Only", "🔄 Both (Text + Speech)"]:
            st.subheader("🔊 Speech Settings")
            if tts_engine:
                try:
                    voices = tts_engine.getProperty('voices')
                    if voices and len(voices) > 1:
                        voice_names = [f"Voice {i+1}" for i in range(len(voices))]
                        selected_voice = st.selectbox("Select Voice", voice_names)
                        voice_index = voice_names.index(selected_voice)
                        tts_engine.setProperty('voice', voices[voice_index].id)
                    else:
                        st.info("Only one voice available")
                    
                    speech_rate = st.slider("Speech Rate", 50, 300, 150)
                    tts_engine.setProperty('rate', speech_rate)
                    
                    volume = st.slider("Volume", 0.0, 1.0, 0.9)
                    tts_engine.setProperty('volume', volume)
                except Exception as e:
                    st.error(f"Settings error: {str(e)}")
            else:
                st.error("TTS engine not available")
            st.markdown("---")
        
        # Status indicators
        st.subheader("📊 Status")
        if model_loaded:
            st.success("✅ Chatbot model loaded")
        else:
            st.error("❌ Chatbot model failed to load")
        
        if tts_engine:
            st.success("✅ Text-to-Speech ready")
        else:
            st.error("❌ Text-to-Speech unavailable")
        
        # Microphone check (for speech modes)
        if input_method in ["🎤 Speech Only", "🔄 Both (Text + Speech)"]:
            try:
                sr.Microphone.list_microphone_names()
                st.success("✅ Microphone available")
            except:
                st.error("❌ Microphone not available")
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📖 How to Use")
        if input_method == "💬 Text Only":
            st.markdown("""
            **Text Mode:**
            • Type your message
            • Press Enter to send
            • Get text responses only
            """)
        elif input_method == "🎤 Speech Only":
            st.markdown("""
            **Speech Mode:**
            • Click 'Start Speaking'
            • Speak clearly
            • Review and send
            • Get text + audio responses
            """)
        else:
            st.markdown("""
            **Combined Mode:**
            • Type OR speak your message
            • Speech auto-fills text
            • Edit before sending
            • Get text + audio responses
            """)

if __name__ == "__main__":
    main()