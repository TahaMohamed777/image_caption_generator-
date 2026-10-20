import io
import html
import wave
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from google import genai
from google.genai import types


TEXT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"
VOICE = "Kore"

# this has to be the first Streamlit command
st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="wide")


# ---------- load everything once (cached, so it doesn't reload on every click) ----------

@st.cache_resource
def load_models():
    # same CNN as in the notebook: VGG16 without the last layer -> 4096 features
    vgg = VGG16()
    cnn = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    # my trained caption model
    model = load_model('best_model.keras')
    # word dictionary and max caption length saved from the notebook
    with open('tokenizer.pkl', 'rb') as f:
        saved = pickle.load(f)
    return cnn, model, saved['word_index'], saved['max_length']


@st.cache_resource
def get_gemini_client():
    # the API key is stored in the app's Secrets on Streamlit Cloud
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


cnn, model, word_index, max_length = load_models()
index_word = {index: word for word, index in word_index.items()}  # number -> word
client = get_gemini_client()


# ---------- my model ----------

def extract_features(image):
    # same preprocessing as the notebook: 224x224, then VGG16 -> 4096 features
    # (load_img in the notebook resizes with "nearest", so I use the same here)
    image = image.convert('RGB').resize((224, 224), Image.NEAREST)
    image = np.array(image, dtype='float32')
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return cnn(image, training=False).numpy()


def text_to_padded_sequence(text):
    # words -> numbers, then pad with zeros at the end (same as pad_sequences with padding='post')
    sequence = [word_index[word] for word in text.split() if word in word_index]
    sequence = sequence[-max_length:]
    return np.array([sequence + [0] * (max_length - len(sequence))])


def predict_caption(feature):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = text_to_padded_sequence(in_text)
        yhat = model([feature, sequence], training=False).numpy()
        word = index_word.get(int(np.argmax(yhat)))
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()


# ---------- Gemini ----------

def improve_caption(caption):
    prompt = (
        "Rewrite this image caption as one clear, natural sentence for a voice assistant "
        "describing a photo to a blind person. Fix the grammar only and do not add any "
        "details that are not in the caption. Return only the sentence.\n\n"
        f"Caption: {caption}"
    )
    try:
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            ),
        )
        return response.text.strip()
    except Exception as e:
        st.warning(f"Gemini couldn't rewrite the caption: {e}")
        return caption


def text_to_speech(text):
    try:
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
                    )
                ),
            ),
        )
        audio_data = response.candidates[0].content.parts[0].inline_data.data
    except Exception as e:
        st.warning(f"Text to speech failed: {e}")
        return None

    # Gemini returns raw audio (16-bit, 24000 Hz, mono), so I wrap it in a wav file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)
    return buffer.getvalue()


# ---------- the web page ----------

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 55%, #F59E0B 100%);
    padding: 2.2rem 2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
}
.hero h1 { color: white; margin: 0; font-size: 2.5rem; }
.hero p { color: rgba(255, 255, 255, 0.92); margin: 0.6rem 0 0; font-size: 1.1rem; }

.step {
    background: #1E293B;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    border-top: 5px solid;
    height: 100%;
}
.step .icon { font-size: 2.2rem; }
.step h4 { margin: 0.4rem 0 0.3rem; }
.step p { opacity: 0.8; margin: 0; font-size: 0.95rem; }

.card {
    background: #1E293B;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1rem;
    border-left: 6px solid;
}
.card .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7; }
.card .text { font-size: 1.35rem; font-weight: 600; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)


def step(icon, title, text, color):
    st.markdown(
        f'<div class="step" style="border-color:{color}"><div class="icon">{icon}</div>'
        f'<h4>{title}</h4><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def card(label, text, color):
    # html.escape so the caption text is shown as text, not read as HTML
    st.markdown(
        f'<div class="card" style="border-color:{color}"><div class="label">{label}</div>'
        f'<div class="text">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown("""
<div class="hero">
    <h1>🖼️ Image Caption Generator</h1>
    <p>Upload a photo and hear what's in it. Made to help blind and visually impaired people.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    step("🧠", "1. CNN (VGG16)", "Looks at the photo and extracts 4096 image features", "#8B5CF6")
with col2:
    step("📝", "2. LSTM", "Writes a caption word by word, trained on Flickr8k", "#EC4899")
with col3:
    step("🔊", "3. Gemini", "Turns it into a natural sentence and reads it aloud", "#F59E0B")

st.write("")
uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    left, right = st.columns(2, gap="large")

    with left:
        st.image(image)

    with right:
        with st.spinner("Generating caption..."):
            caption = predict_caption(extract_features(image))
        card("Caption from my model", caption, "#8B5CF6")

        with st.spinner("Asking Gemini to describe it..."):
            sentence = improve_caption(caption)
        card("Spoken sentence", sentence, "#EC4899")

        with st.spinner("Creating the audio..."):
            audio = text_to_speech(sentence)
        if audio:
            st.audio(audio, format="audio/wav", autoplay=True)

st.markdown("---")
st.caption("NTI project · CNN + LSTM image captioning with Gemini voice output")
