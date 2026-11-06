# 🧠 AI Vision & NLP Dashboard

داشبورد تعاملی Streamlit برای دو مسئله‌ی بینایی ماشین و پردازش زبان طبیعی.

## 🎯 مسائل پروژه

### ۱. Accident Detection (Vision)
تشخیص وقوع تصادف از روی ویدئو با استفاده از ResNet18، LSTM/BiLSTM، Focal Loss و YOLOv8n.

### ۲. Sentiment Analysis (NLP)
تحلیل احساس نظرات محصولات آمازون با پردازش متن، تحلیل سوگیری، Random Undersampling و مدل‌سازی توالی با HMM.

## 🎁 بخش‌های امتیازی

- **ADAS Real-Time Demo**: سیستم استریم زنده با Unidirectional LSTM + YOLO
- **RL Warning System**: تصمیم‌گیری هشدار با DQN (v1 → v3 → v4)
- **HMM Sequence Modeling**: مدل‌سازی توالی متن با Markov Chain

## 👥 تیم

| نقش        | نام                 | ایمیل                       |
| ---------- | ------------------- | --------------------------- |
| Supervisor | Melika Molaei       | melika.molaei@example.com   |
| Student    | Haniyeh Rashidian   | rashidianhaniyeh@gmail.com  |
| Student    | Sina Bahramian      | mohamad.sina1383@gmail.com  |
| Student    | Fateme Bazgir       | Fatemehbzg.fba@gmail.com    |
| Student    | Fateme Shakibamehr  | Fatemeshakiba1045@gmail.com |
| Student    | Pourya Einolghazaei | pourya.civil@gmail.com      |

## 🚀 نصب و اجرا

```bash
pip install -r requirements.txt
cd streamlit
streamlit run app.py