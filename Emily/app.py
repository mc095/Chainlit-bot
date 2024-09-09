import os
from huggingface_hub import InferenceClient
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chainlit as cl
from typing import Optional
from langchain.memory import ConversationBufferMemory
from literalai import LiteralClient

# Load environment variables
load_dotenv()

# Initialize Literal AI Client
literal_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))

# Download NLTK data
nltk.download('vader_lexicon')

# Configure Hugging Face API
client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token=os.getenv("HF_API_KEY"),
)

# Advanced sentiment analysis function
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity

    if sentiment_scores['compound'] >= 0.1:
        overall = 'positive'
    elif sentiment_scores['compound'] <= -0.1:
        overall = 'negative'
    else:
        overall = 'neutral'

    intensity = abs(sentiment_scores['compound'])

    return {
        'overall': overall,
        'intensity': intensity,
        'subjectivity': subjectivity,
        'scores': sentiment_scores
    }
    
SYSTEM_PROMPT_GENERAL = """
You are Ashley, an empathetic AI focused on mental health support. Your goal is to provide personalized, mature, and supportive responses tailored to the user's emotional state, age, and professional background.

Behavior Guidelines:

1. Introduction: Introduce yourself as "Ashley" only during the first interaction.
2. Personalization: Adapt your responses to the user's age and professional background:
   - Offer relatable support for high school students.
   - Provide nuanced advice for professionals.
3. Empathy: Use sentiment analysis to detect emotional cues and respond with genuine encouragement.
4. Evidence-Based Advice: Base your guidance on established psychological research and best practices. If necessary, recommend professional consultation.
5. Self-Reflection: Encourage users to explore their thoughts and emotions with thought-provoking questions.
6. Positive Outlook: Balance acknowledging challenges with guiding users toward constructive solutions.
7. Targeted Support: Address specific concerns:
   - Academic pressure for students.
   - Career stress for professionals.
8. Holistic Wellness: Promote sleep, nutrition, and exercise with practical tips for daily integration.
9. Inspirational Content: Share uplifting stories, practical tips, and occasionally simple recipes for mental well-being.
10. Community Impact: Highlight the positive societal impact of personal development.
11. Topic Focus: Gently redirect off-topic questions (e.g., about places, celebrities, or homework) back to mental health.

Response Style:
    
- Conciseness: Keep your responses brief yet impactful.
- Sentiment Sensitivity: Tailor language and tone to the user's emotional state.
- Important : Direct Focus: Avoid meta-commentary; provide relevant, actionable advice.

Objective:
Deliver thoughtful, supportive guidance that fosters mental well-being and personal growth, staying attuned to each user’s unique needs and challenges.
"""

# Define LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["system_prompt", "user_input", "sentiment"],
    template="{system_prompt}\nUser's emotional state: {sentiment}\nUser: {user_input}\nAshley:"
)

# A simple dictionary to store user passwords (replace with a secure database in production)
user_passwords = {}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    try:
        # Try to get the user from Literal AI
        user_info = literal_client.api.get_user(identifier=username)
        
        # User exists in Literal AI, check if we have a password for them
        if username in user_passwords:
            if user_passwords[username] == password:
                return cl.User(identifier=username, metadata={"role": "user", "info": user_info})
            else:
                print(f"Invalid password for user: {username}")
                return None
        else:
            # User exists in Literal AI but not in our password store
            # We'll treat this as a new user registration
            user_passwords[username] = password
            print(f"Registered existing Literal AI user: {username}")
            return cl.User(identifier=username, metadata={"role": "user", "info": user_info})
    except Exception as e:
        # User doesn't exist in Literal AI, create new user
        try:
            new_user_info = literal_client.api.create_user(identifier=username)
            user_passwords[username] = password
            print(f"Created new user: {username}")
            return cl.User(identifier=username, metadata={"role": "user", "info": new_user_info})
        except Exception as e:
            print(f"Error creating new user: {e}")
            return None

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Morning motivation boost",
            message="I'm feeling stuck and unmotivated this morning. Can you help me identify the reasons behind my lack of motivation and provide some tips to get me moving?",
            icon="/public/coffee-cup.png",
        ),
        cl.Starter(
            label="Stress management techniques",
            message="I'm feeling overwhelmed with stress and anxiety. Can you teach me some effective stress management techniques to help me calm down and focus?",
            icon="/public/sneakers.png",
        ),
        cl.Starter(
            label="Goal setting for mental well-being",
            message="I want to prioritize my mental well-being, but I'm not sure where to start. Can you help me set some achievable goals and create a plan to improve my mental health?",
            icon="/public/meditation.png",
        ),
        cl.Starter(
            label="Building self-care habits",
            message="I know self-care is important, but I struggle to make it a priority. Can you help me identify some self-care activities that I enjoy and create a schedule to incorporate them into my daily routine?",
            icon="/public/idol.png",
        )
    ]

@cl.on_message
async def main(message: cl.Message):
    # Analyze sentiment
    sentiment_info = analyze_sentiment(message.content)
    sentiment_description = f"Sentiment: {sentiment_info['overall']}, Intensity: {sentiment_info['intensity']:.2f}, Subjectivity: {sentiment_info['subjectivity']:.2f}"

    # Prepare the prompt content with the system prompt, sentiment, and user input
    prompt_content = prompt_template.format(
        system_prompt=SYSTEM_PROMPT_GENERAL,
        user_input=message.content,
        sentiment=sentiment_description
    )

    # Send the compiled prompt to the LLM
    response = ""
    msg = cl.Message(content="")
    await msg.send()

    for chunk in client.chat_completion(
        messages=[{"role": "user", "content": prompt_content}],
        max_tokens=500,
        stream=True,
    ):
        token = chunk.choices[0].delta.content
        
        if "Ashley:" in token:
            token = token.split("Ashley:")[1].strip()
        
        if token:
            response += token
            await msg.stream_token(token)

    await msg.update()
     
@cl.on_chat_resume
async def on_chat_resume(thread: cl.ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)