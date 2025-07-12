This project is an **interactive Streamlit app** that predicts shipment **delivery delays** using a **Random Forest Regressor**. It includes **EDA**, model building, and visualizations to help logistics professionals and supply chain managers understand and predict potential delays in their operations.

---

## 🧠 Key Features

- 📥 Upload your own CSV supply chain data
- 🧼 Automatically clean and preprocess data
- 📊 Perform **Exploratory Data Analysis** (EDA) with:
  - Basic stats
  - Correlation heatmaps
  - Distribution plots
- 🔮 Predict delivery delays using **Random Forest Regression**
- 🧪 Model evaluation: MSE, RMSE, R²
- 📉 Visualize:
  - Actual vs Predicted delay scatter plot
  - Feature importance bar chart

---

## ⚙️ Technologies Used

- **Python 3.8+**
- **Streamlit** for the interactive UI
- **scikit-learn** for ML modeling
- **Matplotlib** & **Seaborn** for plotting
- **Pandas / NumPy** for data processing

---

## 📂 Input File Format

Upload a CSV file (`.csv`) with at least the following columns:

- `shipping_date`
- `delivery_date`
- Any other **numeric** features such as quantity, cost, shipping mode, etc.

The app automatically:
- Converts `shipping_date` and `delivery_date` to datetime
- Computes the new target variable `delay_days`

---

## 🧪 How It Works

1. Upload CSV file
2. Automatically clean and parse data
3. Optionally explore with EDA
4. Build Random Forest model to predict delay
5. View model metrics and visualizations

---

## 🛠️ Setup Instructions

### 🔗 Install Dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
🖼️ Screenshots
🔍 EDA Output
Correlation heatmap

Distribution plots

🔮 Model Output
Actual vs. Predicted scatter plot

Feature importance bar chart

🧠 Model Summary
Model: RandomForestRegressor

Target: delay_days

Scaling: StandardScaler on features

Evaluation Metrics:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

📈 Future Improvements
Add support for LSTM-based time series prediction

Dynamic feature engineering from text columns (e.g., shipping mode)

Option to download predictions

Deploy on cloud (Streamlit Cloud or Hugging Face Spaces)

📜 License
MIT License — use freely with attribution.

🙌 Acknowledgments
Streamlit

Scikit-learn

Seaborn

Dataset inspiration: [Supply Chain Sample Data (Kaggle / IBM)]
