import speech_recognition as sr

print(sr.__version__)

"""
    There are 7 methods to recognize speech from an audio source using different APIs.
    1. recognize_bing(): Microsoft Bing Speech
    2. recognize_google(): Google Web Speech API
    3. recognize_google_cloud(): Google Cloud Speech - requires installation of the google-cloud-speech package
    4. recognize_houndify(): Houndify by SoundHound
    5. recognize_ibm(): IBM Speech to Text
    6. recognize_sphinx(): CMU Sphinx - requires installing PocketSphinx
    7. recognize_wit(): Wit.ai

    - Only sphinx works offline. The rest needs an internet connection
    - Only google doesn't need an API key or authentication. The rest of them needs it. Even with google, it's
      not production ready. We can only make 50 queries a day.

    """

# work around incase the audio file cannot be read.
import soundfile
data, samplerate = soundfile.read('path_to_audiofile')
soundfile.write('path_to_audiofile', data, samplerate, subtype='PCM_16')

r = sr.Recognizer()

harvard_audio = sr.AudioFile('path_to_audiofile')
with harvard_audio as source:
    audio = r.record(source) # read the entire audio file into an audio data instance

print(r.recognize_google(audio))