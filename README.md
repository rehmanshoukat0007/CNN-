# Skin Image Classification (CNN)

A Streamlit demo that loads the supplied CNN configuration and weights from `skin_disease_cnn_model.pkl` and predicts one of the two dataset labels: `Cancer` or `Non_cancer`.

## Files needed to run the app

Keep these files together in the root of your GitHub repository:

```text
app.py
requirements.txt
README.md
.gitignore
skin_disease_cnn_model.pkl
```

The training script (`untitled4.py`) and dataset archive (`archive (5).zip`) are not required for image prediction in the deployed app. They may be kept in the repository if desired, but the app does not read them.

## Run locally

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and `skin_disease_cnn_model.pkl` to a GitHub repository.
2. Open https://share.streamlit.io/ and create a new app.
3. Select the repository and branch.
4. Set **Main file path** to `app.py`.
5. Deploy.

## Model details and assumptions

The supplied training script builds a CNN with input size `128 x 128 x 3`, rescales pixels to `[0, 1]`, and trains two output classes. Its `flow_from_dataframe` generator uses alphabetical class ordering by default; for the supplied folders this gives `Cancer` first and `Non_cancer` second. The app uses that same order.

The `.pkl` file contains a Keras model configuration and weights, rather than a complete saved Keras model. The app rebuilds the model from the configuration and then applies the weights. The pickle file must be from a trusted source because loading pickle files can execute code.

## Important medical limitation

This is an educational/research demonstration, not a medical device or diagnostic service. A prediction or confidence score must not be used to diagnose, exclude, or make treatment decisions about skin cancer. Image quality, dataset limitations, and model errors can affect results. Consult a qualified healthcare professional for medical concerns.
