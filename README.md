AI Food Safety Advisor

Project Description

AI Food Safety Advisor is a machine learning-based group project developed as part of the BCA Major Project. It detects food type and evaluates food freshness using deep learning models and intelligent safety logic.

Features

Food detection using CNN (MobileNetV2)
Spoilage detection (Fresh vs Mold)
Safety prediction using time and environment
Streamlit-based interactive UI
Confidence score visualization

Technologies Used

Python
TensorFlow / Keras
CNN Deep Learning
Scikit-learn (Random Forest)
Streamlit
NumPy, Pandas

Project Structure

app.py → Main application (Streamlit UI)
food_model.py → Food detection model
spoilage_model.py → Spoilage detection model
regression.py → Shelf-life prediction
image_downloader.py → Dataset generation
models/ → Saved models (.h5, .keras, .pkl)
dataset/ → Training and testing images
requirements.txt → Dependencies


How to Run

1. Install dependencies
   pip install -r requirements.txt

2. Run the project
   streamlit run app.py

3. Open in browser
   http://localhost:8501

Working Process

1. User uploads food image
2. AI detects food type
3. AI checks spoilage
4. User enters time and storage condition
5. System predicts Safe to Eat, Eat Soon, or Not Safe to Eat

Limitations

Limited dataset size
Accuracy depends on image quality
Some food items may be misclassified


Team Members

Divyansh Mahor (BCAN1CA23144)
Ananya Bhadoria (BCAN1CA23108)
Vikash Pal (BCAN1CA23093)
Aditya Rawat (BCAN1CA23115)
