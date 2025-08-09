 Apple Stock Price Prediction – MLP Neural Network  

Ever wondered if AI could predict Apple’s stock price?  
This project uses a **Multilayer Perceptron (MLP) neural network** to learn from Apple’s past stock data and guess the **closing price** — all wrapped up in an **interactive web app** you can play with! 🚀  

📖 What’s This About?  
We take historical Apple stock data (`AAPL.csv`) — including **Open**, **High**, **Low**, and **Volume** — and run it through:  
1. **Data Preprocessing** – Log-transforming Volume and scaling all features for better learning  
2. **MLP Neural Network** – Two hidden layers (64 & 32 neurons) to capture price patterns  
3. **Interactive Predictions** – Using **Gradio**, you just type in values, click, and boom — your prediction appears with a neat little chart 📊  


✨ Features  
- 🔄 **End-to-End Pipeline** – From raw CSV → trained model → live predictions  
- ⚡ **Instant Results** – No retraining needed thanks to saved `.pkl` files  
- 🎨 **Visual Output** – Side-by-side plot of your Open price vs. predicted Close price  
- 🖱 **No-Code Interface** – Just open the app, type values, and get predictions  


🛠 Tech Stack  
- **Python** 🐍  
- **Pandas, NumPy** – Data wrangling  
- **Scikit-learn** – Model training  
- **Matplotlib** – Visualization  
- **Gradio** – Web interface  

---

 📂 Project Structure  
├── app.py # Gradio web app
├── train_model.py # Model training script
├── AAPL.csv # Historical Apple stock data
├── mlp_model.pkl # Trained MLP model
├── mlp_scaler.pkl # Scaler for preprocessing
└── README.md # This file

🚀 How to Run  
1. Clone this repo  
   ```bash
   git clone https://github.com/YashwanthReddy1406/Stock-Price-Prediction.git
   cd apple-stock-prediction
Install dependencies

*pip install -r requirements.txt
*Run the Gradio app
*python app.py
Open the local link and start predicting 📈

🧠 How It Works (In Simple Words)
Think of the model as a student who has studied years of Apple’s price history.
You give it today’s numbers, and it gives its best guess for the closing price — plus a little graph to show the story visually.
