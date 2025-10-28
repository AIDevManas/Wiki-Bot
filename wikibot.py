import wikipedia
import google.generativeai as genai
import os


def wiki_content(query):
    """Search Wikipedia for a query and return the titles of the top 5 results."""
    print("Loading Wikipedia Chatbot...")
    title_list = wikipedia.search(query, results=5)
    print("Found pages for topic : {title_list}")
    allpage_content = ""
    for pages in title_list:
        if wikipedia.exceptions.PageError:
            pass
        else:
            page = wikipedia.page(pages)
            allpage_content += page.content + "\n"
        return allpage_content


print(
    "Welcome to the Wiki Chatbot! Enter a topic to get started or type 'exit' to quit."
)

history = []
wiki_info = ""
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
while True:
    user_input = input()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    else:
        wiki_info = wiki_content(user_input)

    response = genai.chat.completions.create(
        model="gemini-1.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides information from Wikipedia for content {wiki_info}.",
            },
            *history,
            {"role": "user", "content": user_input},
        ],
    )

    bot_reply = response.choices[0].message["content"]
    print(f"Bot: {bot_reply}")

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_reply})
