import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI Food Safety Advisor", layout="centered")

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f0f0f,#1c1c1c);
color:white;
font-family: 'Segoe UI', sans-serif;
}

.main-title{
text-align:center;
font-size:42px;
font-weight:700;
color:#8b9cff;
}

.subtitle{
text-align:center;
color:#cccccc;
margin-bottom:20px;
}

.card{
background:rgba(255,255,255,0.05);
padding:25px;
border-radius:15px;
margin-top:20px;
}

.safe{
background:rgba(0,255,150,0.12);
padding:12px;
border-radius:10px;
color:#00ffa6;
font-weight:600;
}

.warning{
background:rgba(255,200,0,0.12);
padding:12px;
border-radius:10px;
color:#ffd54a;
font-weight:600;
}

.danger{
background:rgba(255,0,80,0.15);
padding:12px;
border-radius:10px;
color:#ff4d6d;
font-weight:600;
}

/* FIX FILE UPLOADER TEXT */

.stFileUploader label{
color:white !important;
font-weight:600;
}

.stFileUploader div{
color:white !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    food_model = load_model("food_detection_model.keras", compile=False)
    spoilage_model = load_model("food_spoilage_model.h5", compile=False)
    return food_model, spoilage_model

food_model, spoilage_model = load_models()

st.sidebar.title("Project Information")

st.sidebar.write("""
AI system that detects food type and checks food spoilage.

Technologies Used:
• Python
• TensorFlow / Keras
• CNN Deep Learning
• Streamlit Interface

Food Classes:
• Dal
• Rice
• Roti
• Other
""")

st.sidebar.write("Dataset contains food images used to train the AI model.")

st.markdown('<div class="main-title">🥗 AI Food Safety Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect food type and freshness using AI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Food Image", type=["jpg","jpeg","png","webp"])

def detect_food(img):

    img = img.resize((150,150))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = food_model.predict(img_array, verbose=0)

    index = np.argmax(prediction)
    confidence = float(prediction[0][index])

    classes = ["dal","other","rice","roti"]

    if confidence < 0.60:
        food = "other"
    else:
        food = classes[index]

    return food, confidence

def detect_spoilage(img):

    img = img.resize((150,150))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = spoilage_model.predict(img_array, verbose=0)

    prob = float(prediction[0][0])

    if prob > 0.45:
        return "Mold", prob
    else:
        return "Fresh", 1 - prob

def safety_logic(food, hours, env):

    risk = 0
    explanation = []

    if env == "Summer":
        risk += 2
        explanation.append("High temperature increases bacterial growth.")

    elif env == "Room Temperature":
        risk += 1
        explanation.append("Food left at room temperature may spoil faster.")

    elif env == "Winter":
        explanation.append("Cold weather slows bacterial growth.")

    elif env == "Refrigerator":
        risk -= 1
        explanation.append("Refrigeration helps preserve food.")

    if hours > 10:
        risk += 2
        explanation.append("Food has been stored for a long time.")

    elif hours > 6:
        risk += 1
        explanation.append("Food has been sitting for several hours.")

    if food == "rice":
        risk += 1
        explanation.append("Cooked rice can develop Bacillus bacteria.")

    if risk <= 0:
        result = "Safe to Eat"
        explanation.append("Food storage conditions appear safe.")

    elif risk == 1:
        result = "Eat Soon"
        explanation.append("Food should be consumed soon.")

    else:
        result = "Not Safe to Eat"
        explanation.append("High risk of bacterial contamination.")

    return result, explanation

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, width=320)

    food, food_conf = detect_food(img)
    condition, cond_conf = detect_spoilage(img)

  

    st.subheader("AI Analysis")

    st.write(f"Detected Food: **{food}**")
    st.progress(food_conf)

    st.write(f"Food Condition: **{condition}**")
    st.progress(cond_conf)

    if condition == "Mold":

        st.markdown('<div class="danger">❌ Mold detected. Do NOT eat this food.</div>', unsafe_allow_html=True)

        st.write("This food shows visible fungal contamination.")
        st.write("Mold can produce harmful toxins that may cause illness.")
        st.write("Removing mold does not make the food safe.")
        st.write("Recommendation: Discard the food immediately.")

    else:

        st.markdown('<div class="safe">Food appears fresh</div>', unsafe_allow_html=True)

        hours = st.number_input("Hours since cooked",0,48,0)

        env = st.selectbox(
            "Storage Environment",
            ["Room Temperature","Winter","Summer","Refrigerator"]
        )

        if st.button("Check Food Safety"):

            result, explanation = safety_logic(food, hours, env)

            if result == "Safe to Eat":
                st.markdown('<div class="safe">🟢 Safe to Eat</div>', unsafe_allow_html=True)

            elif result == "Eat Soon":
                st.markdown('<div class="warning">⚠ Eat Soon</div>', unsafe_allow_html=True)

            else:
                st.markdown('<div class="danger">❌ Not Safe to Eat</div>', unsafe_allow_html=True)

            st.subheader("Safety Explanation")

            for line in explanation:
                st.write("•", line)

    st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Model Limitations")

st.write("""
• AI accuracy depends on image quality  
• Dataset size is limited  
• Some foods may visually look similar  
• Model may misclassify unfamiliar dishes  
""")