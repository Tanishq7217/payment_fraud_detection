import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#read the file 
with open("myinfo.txt", "r") as f:
    my_info = f.read()

print("Chat with AI! Type 'quit' to exit")

conversation_history = [
    {"role": "system", "content": f"You are my viva guider who will help in clearing eto coc oral examination with its possible following questions also and can detect if there is speliing mistake in my input you will correct it\n\n{my_info}"}
]

while True:
    user_input = input("You: ")

    if user_input == "quit":
        print("AI: Goodbye")
        break

    conversation_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
       model="llama-3.1-8b-instant",
        messages=conversation_history
    )

    ai_reply = response.choices[0].message.content

    conversation_history.append({"role": "assistant", "content": ai_reply})

    print("AI:", response.choices[0].message.content)
    print()