import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from imblearn.over_sampling import SMOTE
import streamlit as st

# โหลดข้อมูล
data = pd.read_csv('loan_data.csv')

# ตรวจสอบข้อมูล
print(data.head())

# กำจัดค่าที่หายไป
data = data.dropna()

# แปลงข้อมูล categorical เป็น numeric
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# แยก features และ target
X = data.drop('not.fully.paid', axis=1)  
y = data['not.fully.paid']

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ SMOTE เพื่อจัดการปัญหาความไม่สมดุลของข้อมูล
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Scaling ข้อมูล
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)  # ใช้ข้อมูลที่สมดุลสำหรับการ Train
X_test = scaler.transform(X_test)

# สร้างโมเดล Neural Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_resampled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile โมเดล
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# เทรนโมเดล
history = model.fit(X_resampled, y_resampled, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# สร้าง Streamlit App
st.title("Loan Prediction App")
st.write("กรอกข้อมูลด้านล่างเพื่อทำนายผล")

# เพิ่มฟีเจอร์ purpose ใน columns_info
columns_info = {
    'credit.policy': ("นโยบายสินเชื่อ (0 = ไม่ผ่าน, 1 = ผ่าน)", "1"),
    'int.rate': ("อัตราดอกเบี้ย (%) เช่น 0.15", "0.15"),
    'installment': ("จำนวนเงินที่ต้องจ่ายต่อเดือน ($)", "250"),
    'log.annual.inc': ("Log รายได้ประจำปี (log($))", "10.5"),
    'dti': ("Debt-to-Income Ratio (%) เช่น 15.0", "15.0"),
    'fico': ("คะแนนเครดิต", "700"),
    'days.with.cr.line': ("จำนวนวันที่มีวงเงินเครดิต (วัน)", "4000"),
    'revol.bal': ("ยอดหนี้หมุนเวียน ($)", "12000"),
    'revol.util': ("เปอร์เซ็นต์การใช้วงเงินหมุนเวียน (%) เช่น 50.0", "50.0"),
    'inq.last.6mths': ("จำนวนการขอสินเชื่อใน 6 เดือนล่าสุด (ครั้ง)", "2"),
    'delinq.2yrs': ("จำนวนครั้งที่ผิดนัดชำระใน 2 ปีล่าสุด (ครั้ง)", "0"),
    'pub.rec': ("จำนวนประวัติการล้มละลาย (ครั้ง)", "0"),
    'purpose': ("วัตถุประสงค์การกู้ยืม (เลือกตามตัวเลือก)", "credit_card")
}

# สร้าง input สำหรับข้อมูลใหม่
input_data = {}
for column, (desc, example) in columns_info.items():
    if column == 'purpose':  # ใช้ selectbox สำหรับ purpose
        input_data[column] = st.selectbox(
            f"{desc}",
            options=label_encoders[column].classes_,
            index=0
        )
    else:
        input_data[column] = st.text_input(f"{desc} (ตัวอย่าง: {example})", value=example)

# แปลง categorical features
for column, le in label_encoders.items():
    if column in input_data:
        input_data[column] = le.transform([input_data[column]])[0]  # แปลงเป็น numeric

# สร้าง DataFrame
input_df = pd.DataFrame([input_data])

# จัดการ columns ที่ขาดหาย
missing_columns = [col for col in X.columns if col not in input_df.columns]
for col in missing_columns:
    input_df[col] = 0  # เติมค่า default สำหรับฟีเจอร์ที่ขาด

# จัดเรียงคอลัมน์ของ input_df ให้ตรงกับ X_train
input_df = input_df[X.columns]

# ตรวจสอบความไม่สมดุลของ Target
from collections import Counter


print("การแบ่งสัดส่วนของข้อมูลก่อนการ SMOTE:")
print(Counter(y))  

# แสดงเป็นเปอร์เซ็นต์
total = len(y)
class_distribution = {k: v / total * 100 for k, v in Counter(y).items()}
print("เป็น %", class_distribution)


print("ารแบ่งสัดส่วนของข้อมูลหลังการ SMOTE:")
print(Counter(y_resampled))  


# ทำนายผล
if st.button("Predict"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probability = prediction[0][0] 
        result = "ค่าความเป็นไปได้ที่ผู้ขอกู้จะสามารถชำระสินเชื่อในเวลาที่กำหนดได้มีค่าสูง อนุมัติสินเชื่อ" if probability > 0.5 else "ค่าความเป็นไปได้ที่ผู้ขอกู้จะสามารถชำระสินเชื่อในเวลาที่กำหนดได้มีค่าต่ำ ไม่อนุมัติสินเชื่อ"
        st.write(f"ผลการทำนาย: {result}")
        st.write(f"ค่าความน่าจะเป็น: {probability:.2f}")
    except Exception as e:
        st.write("Error:", e)
