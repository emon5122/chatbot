import datetime
import webbrowser
import nltk
import openai
import pyttsx3
import requests
import speech_recognition as sr
import tensorflow as tf


def speak(voice):
  engine = pyttsx3.init()
  rate = engine.getProperty('rate')
  engine.setProperty('rate', rate + 1)
  engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Alex')
  engine.say(voice)
  engine.runAndWait()


# download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# use NLTK's word tokenization function

# use NLTK's part-of-speech tagging function

# use NLTK's legitimization function
from nltk.stem import WordNetLemmatizer

# create a WordNet legitimatize object
lemmatizer = WordNetLemmatizer()


# Send a GET request to the Time API
response = requests.get("http://worldtimeapi.org/api/timezone/America/New_York")

# Get the current date and time from the API response
date_time = response.json()["datetime"]

# Print the date and time
print(date_time)

current_time = datetime.datetime.now().time()
current_date = datetime.datetime.now().date()


# Set up the OpenAI API client
openai.api_key = "ggggggggggggg"

# Set the model to use
model_engine = "text-davinci-003"

# Set the trigger phrase
trigger_phrase = "Hey Sophia"

openai.api_key = 'ggggghhhhhhhhhhh'

engine = pyttsx3.init()

r = sr.Recognizer()
mic = sr.Microphone(device_index=1)

conversation = ""
user_name = "Steve"
bot_name = "Sophia"

# Set up an empty string to store the conversation
conversation = ""

# Get the list of available voices
voices = engine.getProperty('voices')


# Set the voice to a specific female voice by specifying the voice name
for voice in voices:
    if voice.name == 'Microsoft Zira Desktop - English (United States)':
        engine.setProperty('voice', voice.id)
        break


# Set the speaking rate to a slower rate
engine.setProperty('rate', -5)

# Set the speaking rate to a faster rate
# engine.setProperty('rate', 5)


async def fetch_response(conversation):
  # fetch response from open AI api
  response = openai.Completion.create(engine='text-davinci-003', prompt=conversation, max_tokens=100)
  response_str = response["choices"][0]["text"].replace("\n", "")
  response_str = response_str.split(f"{user_name}: ", 1)[0].split(
      f"{bot_name}: ", 1)[0]
  return response_str


while True:
    with mic as source:
        print("\nlistening...")
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio = r.listen(source)
    print("no longer listening.\n")

    try:
        user_input = r.recognize_google(audio)
    except:
        continue

    prompt = user_name + ": " + user_input + "\n" + bot_name + ": "

    conversation += prompt  # allows for context

    # fetch response from open AI api
    response = openai.Completion.create(engine='text-davinci-003', prompt=conversation, max_tokens=100)
    response_str = response["choices"][0]["text"].replace("\n", "")
    response_str = response_str.split(user_name + ": ", 1)[0].split(bot_name + ": ", 1)[0]

    # Continue with the OpenAI response as usual
    prompt = user_name + ": " + user_input + "\n Sophia:"

    # Define the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.09), loss='mean_squared_error')

    # Provide the model with some training data
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]

    # Train the model
    model.fit(x, y, epochs=5)

    # Use the model to make predictions
    predictions = model.predict([10, 20, 30, 40])
    print(predictions)


    # Define a function to generate responses using GPT-3

    def generate_response(prompt):
      completions = openai.Completion.create(
          prompt=prompt,
          max_tokens=3771,
          n=1,
          stop=None,
          temperature=0.7,
      )
      return completions.choices[0].text


    conversation += response_str + "\n"
    print(response_str)

    engine.say(response_str)
    engine.runAndWait()

    # Setting up the conversation
    print("Hello! How can I help you?")

    # Getting the user's input
    user_input = input()

    # Checking to see if the user wants to open a web browser
    if 'open' in user_input and 'browser' in user_input:
        webbrowser.open('http://www.google.com')
        print('Browser opened!')

    # Checking to see if the user wants to open YouTube
    elif 'YouTube' in user_input:
        webbrowser.open('https://www.youtube.com')
        print('YouTube opened!')

    # Checking to see if the user wants to search for a video on YouTube
    elif 'search' in user_input and 'video' in user_input:
        # Getting the video title from the user
        video_title = input("What video do you want to search for? ")

        # Searching YouTube for the video
        query_string = {'search_query': video_title}
        r = requests.get('http://www.youtube.com/results', params=query_string)

        # Pulling up the first video on the results page
        search_results = r.text
        video_id = search_results.split('href="https://www.youtube.com/watch?v=')[1].split('"')[0]


        def humanize_chatbot(chatbot):
          # Use Personality Forge API to add human-like features to your chatbot
          personality_forge_api = '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'

          # Set the chatbots personality
          personality_id = 'YOUR PERSONALITY ID HERE'

            # Use the API to create a human-like personality for your chatbot
          response = requests.post(
              f'https://www.personalityforge.com/api/chat/?apiKey={personality_forge_api}&message={chatbot.message}&chatBotID={personality_id}'
          )

          # Update the chatbot's response with the human-like response
          chatbot.message = response.json()['message']

        # Opening the video
        webbrowser.open('https://www.youtube.com/watch?v=' + video_id)
        print('Video opened!')
#test