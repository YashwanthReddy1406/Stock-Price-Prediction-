import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# Load model and scaler
model = joblib.load("mlp_model.pkl")
scaler = joblib.load("mlp_scaler.pkl")

def predict_mlp(open_price, high, low, volume):
    log_volume = np.log1p(volume)
    input_data = np.array([[open_price, high, low, log_volume]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Plot result
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [open_price, prediction], marker='o', color='purple')
    plt.xticks([0, 1], ['Open', 'Predicted Close'])
    plt.title("MLP Predicted Close Price")
    plt.ylabel("Price")
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name)
    plt.close()

    return f"${prediction:.2f}", tmpfile.name

# Gradio app
gr.Interface(
    fn=predict_mlp,
    inputs=[
        gr.Number(label="Open Price"),
        gr.Number(label="High"),
        gr.Number(label="Low"),
        gr.Number(label="Volume")
    ],
    outputs=[
        gr.Textbox(label="Predicted Close Price"),
        gr.Image(type="filepath", label="Plot")
    ],
    title="Apple Stock Predictor (MLP)",
    description="Predict Apple stock close price using a neural network (MLP)."
).launch()
