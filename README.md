# Image Caption Generator with Voice Output

Upload a photo and the app describes it out loud, to help blind and visually impaired people understand photos.

## How it works
1. **CNN (InceptionV3)** turns the photo into 2048 image features
2. **LSTM** model trained on Flickr8k generates a caption word by word
3. **Gemini** rewrites the caption as a natural sentence and reads it aloud

## Files
- `image_caption_generator_improved.ipynb`: training notebook (data cleaning, feature extraction, model, training, BLEU evaluation)
- `streamlit_app.py`: the web app
- `best_model.keras`: the trained caption model
- `tokenizer.pkl`: word dictionary and max caption length used by the model
