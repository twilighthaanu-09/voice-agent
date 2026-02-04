import azure.cognitiveservices.speech as speechsdk
from config import SPEECH_KEY, SPEECH_REGION


def speech_to_text():
    """
    Captures audio from default microphone and converts speech to text.
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    audio_config = speechsdk.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    print("🎤 Speak now...")
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("❌ No speech could be recognized.")
        return None

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.CancellationDetails(result)
        print(f"❌ Speech recognition canceled: {cancellation.reason}")
        return None

    return None


def text_to_speech(text: str):
    """
    Converts text to speech and plays it back.
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config
    )

    synthesizer.speak_text_async(text).get()
