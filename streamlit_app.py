import io
import os
import html
import wave
import pickle
import hashlib
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from google import genai
from google.genai import types


TEXT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"

# this has to be the first Streamlit command
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def text_to_speech(text, voice, slow):
    # the TTS model follows simple instructions about how to speak
    if slow:
        text = "Read this slowly and clearly: " + text
    try:
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
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


# ---------- settings (sidebar) ----------

def sidebar_section(icon, title):
    st.markdown(f'<div class="side-section">{icon} {title}</div>', unsafe_allow_html=True)


with st.sidebar:
    # filled in further down, after the logos are defined
    brand_slot = st.empty()

    # label_visibility="collapsed" hides the small label (the section title shows instead),
    # screen readers still read it
    sidebar_section("🎨", "Display")
    # buttons side by side, one is always selected
    theme_name = st.segmented_control(
        "Display mode",
        ["☀️ Light", "🌙 Dark", "⚫ High contrast"],
        default="🌙 Dark",
        # shorter text on the buttons so all three fit in one row
        format_func=lambda mode: mode.replace("High contrast", "Contrast"),
        label_visibility="collapsed",
    )
    if theme_name is None:  # clicking the selected button again un-selects it, so fall back to Dark
        theme_name = "🌙 Dark"

    sidebar_section("🔠", "Text size")
    # a slider from small to big
    text_size = st.select_slider(
        "Text size",
        options=["Normal", "Large", "Extra large"],
        value="Large",
        label_visibility="collapsed",
    )

    sidebar_section("🔊", "Voice")
    # dropdown list with a short description of each voice
    VOICES = {"Kore": "Firm", "Puck": "Upbeat", "Charon": "Informative", "Aoede": "Breezy"}
    voice = st.selectbox(
        "Voice",
        list(VOICES),
        format_func=lambda name: f"🗣️ {name} · {VOICES[name]}",
        label_visibility="collapsed",
    )
    slow = st.toggle("🐢 Speak slowly")
    hands_free = st.toggle(
        "🙌 Hands-free mode",
        help="Describes the photo and reads it aloud as soon as a photo is added, no buttons needed.",
    )

    sidebar_section("🖼️", "Background")
    show_bg_image = st.toggle(
        "Background image",
        value=True,
        help="Turn this off if the background makes the text hard to read. "
             "It is always off in High contrast mode.",
    )

    footer_slot = st.empty()


# ---------- colors for each display mode ----------

# everything uses blues: dark navy -> light blue, like the logo card
THEMES = {
    "☀️ Light": {
        "bg": "linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #E0F2FE 100%)",
        "card": "#FFFFFF",
        "text": "#0F172A",
        "muted": "#1E3A8A",
        "accent": "#1D4ED8",
        "border": "#93C5FD",
        "sidebar": "rgba(239, 246, 255, 0.88)",
        "overlay": "rgba(255, 255, 255, 0.55)",
        "hero_bg": "linear-gradient(135deg, #0B1E5B 0%, #1D4ED8 55%, #60A5FA 100%)",
        "hero_text": "#FFFFFF",
        "button_bg": "linear-gradient(135deg, #0B1E5B 0%, #1D4ED8 60%, #3B82F6 100%)",
        "button_text": "#FFFFFF",
        # sidebar controls (display buttons, voice dropdown)
        "control_bg": "linear-gradient(135deg, #1E3A8A, #2563EB)",
        "control_active_bg": "linear-gradient(135deg, #1D4ED8, #60A5FA)",
        "control_text": "#FFFFFF",
        "control_active_text": "#FFFFFF",
        "control_border": "#1E40AF",
        "control_ring": "#0B1E5B",
    },
    "🌙 Dark": {
        "bg": "radial-gradient(circle at top left, #1E3A8A 0%, #0F172A 50%, #020617 100%)",
        "card": "#0F1E3D",
        "text": "#F8FAFC",
        "muted": "#BFDBFE",
        "accent": "#60A5FA",
        "border": "#1E3A8A",
        "sidebar": "rgba(8, 15, 40, 0.9)",
        "overlay": "rgba(2, 6, 23, 0.40)",
        "hero_bg": "linear-gradient(135deg, #0A1747 0%, #1E40AF 55%, #3B82F6 100%)",
        "hero_text": "#FFFFFF",
        "button_bg": "linear-gradient(135deg, #0A1747 0%, #1D4ED8 60%, #3B82F6 100%)",
        "button_text": "#FFFFFF",
        "control_bg": "linear-gradient(135deg, #0B1E5B, #1E40AF)",
        "control_active_bg": "linear-gradient(135deg, #1D4ED8, #60A5FA)",
        "control_text": "#DBEAFE",
        "control_active_text": "#FFFFFF",
        "control_border": "#1E3A8A",
        "control_ring": "#93C5FD",
    },
    "⚫ High contrast": {
        "bg": "#000000",
        "card": "#000000",
        "text": "#FFFFFF",
        "muted": "#FFFF00",
        "accent": "#FFFF00",
        "border": "#FFFF00",
        "sidebar": "#000000",
        "overlay": "#000000",
        "hero_bg": "#000000",
        "hero_text": "#FFFF00",
        "button_bg": "#FFFF00",
        "button_text": "#000000",
        "control_bg": "#000000",
        "control_active_bg": "#FFFF00",
        "control_text": "#FFFF00",
        "control_active_text": "#000000",
        "control_border": "#FFFF00",
        "control_ring": "#FFFFFF",
    },
}
FONT_SIZES = {"Normal": "18px", "Large": "21px", "Extra large": "25px"}

theme = THEMES[theme_name]

# the chosen colors become CSS variables, the rest of the CSS below uses them
css_variables = "".join(f"--{name.replace('_', '-')}: {value};" for name, value in theme.items())
st.markdown(
    f"<style>:root {{ {css_variables} --font-size: {FONT_SIZES[text_size]}; }}</style>",
    unsafe_allow_html=True,
)

# background image: the file is in the "static" folder, Streamlit serves it at app/static/...
# a see-through color layer on top keeps the text readable
BACKGROUND_IMAGE = "static/background.jpg"
if show_bg_image and theme_name != "⚫ High contrast" and os.path.exists(BACKGROUND_IMAGE):
    st.markdown("""
<style>
/* the page itself is see-through, the image is drawn on a layer behind everything */
.stApp { background: transparent !important; isolation: isolate; }

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    /* color layer + image; 100% 100% stretches the image so the WHOLE picture fills the screen */
    background: linear-gradient(var(--overlay), var(--overlay)),
                url("app/static/background.jpg") center / 100% 100% no-repeat;
    /* blur layer on top of the image */
    filter: blur(2px);
    /* a little zoom so the blurred edges don't show a white border */
    transform: scale(1.02);
}

/* frosted glass sidebar */
[data-testid="stSidebar"] { backdrop-filter: blur(12px); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
html { font-size: var(--font-size); }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 3px solid var(--border);
    backdrop-filter: blur(12px);
    box-shadow: 6px 0 30px rgba(0, 0, 0, 0.25);
}

/* sidebar: app name card at the top */
.side-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: var(--hero-bg);
    border: 2px solid var(--border);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.side-brand .app-logo { width: 2.8rem; height: 2.8rem; }
.side-brand .name { color: var(--hero-text); font-weight: 800; font-size: 1.1rem; line-height: 1.2; }
.side-brand .sub { color: var(--hero-text); opacity: 0.85; font-size: 0.8rem; }

/* sidebar: section titles */
.side-section {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.3rem 0 0.5rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid var(--border);
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}


/* sidebar: display buttons in blue, the chosen one is lighter with a ring */
[data-testid="stSidebar"] [data-testid^="stBaseButton-segmented_control"] {
    flex: 1 1 0;
    min-width: 0;
    background: var(--control-bg) !important;
    border: 2px solid var(--control-border) !important;
    padding: 0.45rem 0.3rem;
}
[data-testid="stSidebar"] [data-testid^="stBaseButton-segmented_control"] * {
    color: var(--control-text) !important;
    font-weight: 700;
    font-size: 0.85rem;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-segmented_controlActive"] {
    background: var(--control-active-bg) !important;
    box-shadow: 0 0 0 2px var(--control-ring);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-segmented_controlActive"] * {
    color: var(--control-active-text) !important;
}

/* sidebar: voice dropdown in blue */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--control-bg) !important;
    border: 2px solid var(--control-border) !important;
    border-radius: 12px;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--control-text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--control-text) !important; }
/* the dropdown has its own outline, so no extra yellow focus line inside it */
[data-testid="stSidebar"] [data-baseweb="select"] *:focus-visible { outline: none !important; }
[data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div { box-shadow: 0 0 0 3px #FACC15; }

/* sidebar: footer */
.side-footer {
    margin-top: 1.8rem;
    padding-top: 1rem;
    border-top: 2px solid var(--border);
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
}
.side-footer .gemini-badge { margin-bottom: 0.6rem; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

/* big, easy to hit buttons */
.stButton > button {
    width: 100%;
    min-height: 4.5rem;
    background: var(--button-bg);
    border: 3px solid var(--border);
    border-radius: 18px;
    margin-bottom: 0.4rem;
}
.stButton > button p { color: var(--button-text) !important; font-size: 1.4rem; font-weight: 800; }
.stButton > button:hover { filter: brightness(1.12); border-color: var(--text); }

/* thick yellow outline so keyboard users always see where they are */
.stApp *:focus-visible { outline: 4px solid #FACC15 !important; outline-offset: 3px; }

/* upload box */
[data-testid="stFileUploaderDropzone"] {
    background: var(--card);
    border: 3px dashed var(--accent);
    border-radius: 18px;
    padding: 1.6rem;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }

/* header banner */
.hero {
    background: var(--hero-bg);
    color: var(--hero-text);
    border: 3px solid var(--border);
    border-radius: 22px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.hero-title { display: flex; align-items: center; gap: 1.1rem; }
.hero h1 { color: var(--hero-text) !important; margin: 0; padding: 0; font-size: 2.3rem; line-height: 1.15; }
.hero p { color: var(--hero-text) !important; margin: 0.4rem 0 0; font-size: 1.15rem; opacity: 0.95; }
.app-logo { width: 4.2rem; height: 4.2rem; flex-shrink: 0; }

/* "Powered by Gemini" badge */
.gemini-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #FFFFFF;
    color: #1F1F1F;
    border-radius: 999px;
    padding: 0.5rem 1.1rem;
    font-weight: 700;
    font-size: 1rem;
    border: 2px solid var(--border);
}
.gemini-logo { width: 1.5rem; height: 1.5rem; }

/* result cards */
.card {
    background: var(--card);
    border: 3px solid var(--border);
    border-left: 10px solid var(--accent);
    border-radius: 18px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}
.card .label {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted);
}
.card .text { font-size: 1.6rem; font-weight: 700; color: var(--text); margin-top: 0.4rem; line-height: 1.35; }
</style>
""", unsafe_allow_html=True)


# ---------- logos (drawn as SVG so no image files are needed) ----------

# app logo: an eye, because the app "sees" the photo for the user
APP_LOGO = (
    '<svg class="app-logo" viewBox="0 0 64 64" aria-hidden="true">'
    '<circle cx="32" cy="32" r="31" fill="#FFFFFF"/>'
    '<path d="M10 32 C20 16 44 16 54 32 C44 48 20 48 10 32Z" fill="none" stroke="#1D4ED8" stroke-width="4.5"/>'
    '<circle cx="32" cy="32" r="8.5" fill="#0EA5E9"/>'
    '</svg>'
)

# Gemini's four-pointed star with its blue-purple-pink gradient
GEMINI_LOGO = (
    '<svg class="gemini-logo" viewBox="0 0 24 24" aria-hidden="true">'
    '<defs><linearGradient id="gem" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#4285F4"/><stop offset="0.55" stop-color="#9B72CB"/>'
    '<stop offset="1" stop-color="#D96570"/></linearGradient></defs>'
    '<path d="M12 0C12 6.6 17.4 12 24 12C17.4 12 12 17.4 12 24C12 17.4 6.6 12 0 12C6.6 12 12 6.6 12 0Z" fill="url(#gem)"/>'
    '</svg>'
)


def card(icon, label, text):
    # role="status" makes screen readers read the new text out loud when it appears
    # html.escape so the caption is shown as text, not read as HTML
    st.markdown(
        f'<div class="card" role="status" aria-live="polite">'
        f'<div class="label">{icon} {label}</div>'
        f'<div class="text">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


# fill the top and bottom of the sidebar now that the logos exist
brand_slot.markdown(
    f'<div class="side-brand">{APP_LOGO}<div>'
    '<div class="name">Image Caption<br>Generator</div>'
    '<div class="sub">⚙️ Settings</div>'
    '</div></div>',
    unsafe_allow_html=True,
)
footer_slot.markdown(
    f'<div class="side-footer"><div class="gemini-badge">{GEMINI_LOGO} Powered by Gemini</div>'
    '<div>CNN + LSTM · NTI project</div></div>',
    unsafe_allow_html=True,
)


# ---------- the page ----------

st.markdown(
    '<div class="hero">'
    f'<div class="hero-title">{APP_LOGO}<div>'
    '<h1>Image Caption Generator</h1>'
    '<p>Add a photo and hear what is in it. Made for blind and visually impaired people.</p>'
    '</div></div>'
    f'<div class="gemini-badge">{GEMINI_LOGO} Powered by Gemini</div>'
    '</div>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How it works"):
    st.markdown(
        "1. 🧠 **CNN (VGG16)** looks at the photo and turns it into 4096 image features\n"
        "2. 📝 **LSTM** writes a caption word by word (trained on the Flickr8k dataset)\n"
        "3. 🔊 **Gemini** turns the caption into a natural sentence and reads it aloud"
    )

# only show the camera when it's chosen, so the browser doesn't ask for camera access on every visit
source = st.radio("How do you want to add a photo?", ["📁 Upload a photo", "📷 Take a photo"], horizontal=True)
if source == "📁 Upload a photo":
    photo = st.file_uploader("Choose a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
else:
    photo = st.camera_input("Take a photo with your camera")

if photo is None:
    card("👋", "Start here", "Upload a photo or take one with your camera.")
else:
    photo_bytes = photo.getvalue()

    # a new photo clears the old results
    photo_id = hashlib.md5(photo_bytes).hexdigest()
    if st.session_state.get("photo_id") != photo_id:
        st.session_state.photo_id = photo_id
        st.session_state.caption = None
        st.session_state.sentence = None
        st.session_state.audio = None

    image = Image.open(io.BytesIO(photo_bytes))
    left, right = st.columns(2, gap="large")

    with left:
        st.image(image, caption="Your photo")

    with right:
        describe_clicked = st.button("📝 Describe the photo")
        read_clicked = st.button("🔊 Read it aloud")

        # describing is needed before reading aloud, so either button (or hands-free) does it
        if (describe_clicked or read_clicked or hands_free) and st.session_state.caption is None:
            with st.spinner("Looking at the photo..."):
                st.session_state.caption = predict_caption(extract_features(image))
            with st.spinner("Gemini is writing the sentence..."):
                st.session_state.sentence = improve_caption(st.session_state.caption)

        # "Read it aloud" always makes new audio (so a new voice setting is used),
        # hands-free only makes it once per photo
        if read_clicked or (hands_free and st.session_state.audio is None):
            with st.spinner("Creating the voice..."):
                # False (not None) on failure, so hands-free doesn't keep retrying
                st.session_state.audio = text_to_speech(st.session_state.sentence, voice, slow) or False

        if st.session_state.caption:
            card("🧠", "Caption from my model", st.session_state.caption)
            card(GEMINI_LOGO, "Description by Gemini", st.session_state.sentence)

        if st.session_state.audio:
            st.audio(st.session_state.audio, format="audio/wav", autoplay=True)

st.markdown("---")
st.caption("NTI project · CNN (VGG16) + LSTM image captioning · voice by Google Gemini")
