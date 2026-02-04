from speech import speech_to_text, text_to_speech
from llm import get_llm_response


def main():
    print("🧠 Voice Agent POC started (Ctrl+C to exit)")

    while True:
        user_text = speech_to_text()

        if not user_text:
            print("❌ Could not recognize speech, try again...")
            continue

        print(f"👤 User said: {user_text}")

        # Exit condition
        if "bye" in user_text.lower() or "exit" in user_text.lower():
            goodbye_msg = "Goodbye! Take care."
            print(f"🤖 Agent: {goodbye_msg}")
            text_to_speech(goodbye_msg)
            break

        # LLM response
        response = get_llm_response(user_text)
        print(f"🤖 Agent: {response}")

        # Speak response
        text_to_speech(response)


if __name__ == "__main__":
    main()
