import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="AI Vision & NLP Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Vazirmatn', 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #0F172A 50%, #1E1B4B 100%);
        color: #F8FAFC;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    .hero-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }

    .danger-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border-left: 5px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        color: #FEE2E2;
    }
    
    .safe-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border-left: 5px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        color: #D1FAE5;
    }

    .bonus-card {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(126, 34, 206, 0.3) 100%);
        border: 1px solid rgba(192, 132, 252, 0.4);
        border-left: 6px solid #C084FC;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        color: #F3E8FF;
    }

    .warning-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.25) 100%);
        border-left: 5px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        color: #FEF3C7;
    }

    .team-chip {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADERS & DICTIONARIES
# ==========================================
SUPERVISOR = {
    "name": "Melika Molaei",
    "role": "Project Supervisor",
    "email": "melika.molaei@example.com"
}

MEMBERS = [
    {"name": "Haniyeh Rashidian",   "role": "Student", "email": "rashidianhaniyeh@gmail.com"},
    {"name": "Sina Bahramian",      "role": "Student", "email": "mohamad.sina1383@gmail.com"},
    {"name": "Fateme Bazgir",       "role": "Student", "email": "Fatemehbzg.fba@gmail.com"},
    {"name": "Fateme Shakibamehr",  "role": "Student", "email": "Fatemeshakiba1045@gmail.com"},
    {"name": "Pourya Einolghazaei", "role": "Student", "email": "pourya.civil@gmail.com"},
]

@st.cache_data
def load_all_datasets():
    acc_weather = {'Clear': 445, 'Cloudy': 243, 'Rain': 60, 'Unknown': 1, 'Snow': 1}
    acc_light = {'Normal': 680, 'Twilight': 41, 'Dark': 22, 'Bright': 7}
    acc_scene = {'Urban': 475, 'Highway': 171, 'Sub-urban': 75, 'Other': 22, 'Rural': 3, 'Industrial': 3, 'Nature': 1}

    nlp_rating_compare = {'All Reviews (همه نظرات)': 4.021, 'Warranty Reviews (نظرات گارانتی)': 3.194}
    nlp_coverage = {'>= 1': 10050, '>= 2': 3090, '>= 3': 1524, '>= 5': 598, '>= 10': 175}
    nlp_dist = {'1 Star': 28.5, '2 Stars': 11.1, '3 Stars': 10.0, '4 Stars': 13.3, '5 Stars': 37.1}

    top_products = {
        'iPhone Charger Syncwire Lightning Cable': 5.0,
        'Anker Tempered Glass Screen Protector': 4.82,
        'Anker iPhone Charger, Powerline Lightning': 4.76,
        'Belkin 6-Outlet Pivot-Plug Wall Mount': 4.75,
        'Solo Conquer 15.6 Inch Expandable Laptop': 4.75,
        'Anker SoundCore Sport XL Outdoor': 4.69,
        'Bluetooth Headphones TaoTronics Wireless': 4.63,
        'Bluedio T3 Extra Bass Bluetooth Headphones': 4.60,
        'Arteck HB030B Universal Slim Portable': 4.60,
        'Tripp Lite Isobar 2 Outlet Surge Protector': 4.60
    }

    bottom_products = {
        'Amcrest ProHD 720P 4CH Video System': 1.69,
        'Amcrest ProHD 1080P 4CH Video Security': 1.69,
        'Samsung Gear Fit2- Pink, Medium/Large': 1.69,
        'Microsoft Band 2 - Medium': 1.69,
        'Roku 3 Streaming Media Player': 1.62,
        'Seagate Backup Plus Slim 2TB Portable': 1.44,
        'VIZIO E500i-B1 50-Inch 1080p Smart TV': 1.35,
        'Lexar JumpDrive S45 16GB USB 3.0': 1.33,
        'Netgear Nighthawk AC1900 Dual Router': 1.29,
        'Lexar JumpDrive S45 128GB USB 3.0': 1.10
    }

    return {
        'acc_weather': acc_weather,
        'acc_light': acc_light,
        'acc_scene': acc_scene,
        'nlp_rating_compare': nlp_rating_compare,
        'nlp_coverage': nlp_coverage,
        'nlp_dist': nlp_dist,
        'top_products': top_products,
        'bottom_products': bottom_products
    }

data = load_all_datasets()

COLOR_PALETTE = {
    'Weather': {'Clear': '#38BDF8', 'Cloudy': '#94A3B8', 'Rain': '#1E40AF', 'Snow': '#E2E8F0', 'Unknown': '#64748B'},
    'Light': {'Normal': '#06B6D4', 'Twilight': '#F59E0B', 'Dark': '#334155', 'Bright': '#FACC15'},
    'Scene': {'Urban': '#475569', 'Highway': '#B45309', 'Sub-urban': '#64748B', 'Other': '#94A3B8', 'Rural': '#10B981', 'Industrial': '#78350F', 'Nature': '#22C55E'}
}

# Base directory for local assets (videos, images)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h2 style="color: #38BDF8; font-weight: 800; margin: 0;">🧠 Vision & NLP</h2>
    <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Interactive Project Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page_selection = st.sidebar.radio(
    "📍 Select Dashboard Module:",
    ["🏠 Home", "👁️ Vision Pipeline & Iterations", "💬 NLP & Customer Sentiment"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("🚀 Academic AI Project Dashboard")

# ==========================================
# PAGE 0: HOME
# ==========================================
if page_selection == "🏠 Home":
    st.markdown("""
    <div class="hero-card" style="direction: rtl; text-align: right;">
        <h1 style="color: #38BDF8; margin: 0; font-weight: 800;">🏠 AI Vision & NLP Project Dashboard</h1>
        <p style="color: #CBD5E1; font-size: 1.1rem; margin-top: 8px;">
        داشبورد تعاملی دو مسئله‌ی بینایی ماشین (تشخیص تصادف از ویدئو) و پردازش زبان طبیعی (تحلیل احساس نظرات محصولات)
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_sup, col_team = st.columns([1, 2])
    with col_sup:
        st.markdown(f"""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <h3 style="color: #38BDF8; margin-top:0;">🎓 Supervisor & Mentor</h3>
            <p style="font-size: 1.2rem; font-weight: bold; margin-bottom: 2px;">{SUPERVISOR['name']}</p>
            <p style="color: #94A3B8; font-size: 0.9rem;">{SUPERVISOR['role']}</p>
            <p style="color: #38BDF8; font-size: 0.85rem;">📧 {SUPERVISOR['email']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_team:
        team_html = "".join([
            f"<div class='team-chip'><b>{m['name']}</b><br><span style='font-size:0.8rem; color:#94A3B8;'>📧 {m['email']}</span></div>"
            for m in MEMBERS
        ])
        st.markdown(f"""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <h3 style="color: #10B981; margin-top:0;">👥 Team Members</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                {team_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border-right: 4px solid #38BDF8;">
            <h2 style="color: #38BDF8; margin-top: 0;">👁️ Problem 1: Accident Detection (Vision)</h2>
            <p style="font-size: 0.95rem; line-height: 1.8;">
            تشخیص وقوع تصادف از روی ویدئو، مسئله‌ای کاربردی در بینایی ماشین است که می‌تواند زمان اطلاع‌رسانی و اعزام امداد را کاهش دهد.
            در این پروژه با تکیه بر مجموعه‌داده‌ی Nexar (کلیپ‌های کوتاه چند ده ثانیه‌ای) مدلی طراحی شد که با مشاهده‌ی هر ویدئو وقوع تصادف را پیش‌بینی می‌کند.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border-right: 4px solid #10B981;">
            <h2 style="color: #10B981; margin-top: 0;">💬 Problem 2: Sentiment Analysis (NLP)</h2>
            <p style="font-size: 0.95rem; line-height: 1.8;">
            تحلیل احساس شاخه‌ای از پردازش زبان طبیعی است که با استفاده از الگوریتم‌های یادگیری ماشین به شناسایی و استخراج خودکار احساسات پشت متن می‌پردازد.
            در این پروژه با مجموعه‌داده‌ای از نظرات ثبت‌شده برای کالاهای الکترونیکی آمازون، مدلی برای پیش‌بینی امتیاز کاربر (۱ تا ۵) از روی متن نظر ساخته شد.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="direction: rtl; text-align: right;">
        <h3 style="color: #38BDF8; margin-top: 0;">📌 How to Use This Dashboard</h3>
        <p style="font-size: 0.95rem; line-height: 1.8;">
        از منوی سمت راست می‌توانید بین بخش‌های زیر جابه‌جا شوید:
        <br>• <b>👁️ Vision Pipeline & Iterations</b>: بررسی تحلیل داده‌ها، سوگیری انتخاب، شش آزمایش مدل‌سازی و سیستم استریم زنده.
        <br>• <b>💬 NLP & Customer Sentiment</b>: بررسی پنج نمودار تحلیلی و روندهای نظرات محصولات الکترونیکی.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 1: VISION
# ==========================================
elif page_selection == "👁️ Vision Pipeline & Iterations":
    st.markdown("""
    <div class="hero-card" style="direction: rtl; text-align: right;">
        <h1 style="color: #38BDF8; margin: 0; font-weight: 800;">👁️ Vision Pipeline & Iterations</h1>
        <p style="color: #CBD5E1; font-size: 1.1rem; margin-top: 8px;">
        تحلیل داده‌ها، بررسی سوگیری، و شش آزمایش مدل‌سازی عمیق تا رسیدن به مدل نهایی
        </p>
    </div>
    """, unsafe_allow_html=True)

    vis_tab1, vis_tab2, vis_tab3, vis_tab4, vis_tab5 = st.tabs([
    "📊 1. Exploratory Data Analysis (EDA)",
    "⚖️ 2. Selection Bias Analysis",
    "🧪 3. Model Iterations (Try 1 to 6)",
    "🚀 4. ADAS Real-Time Demo (Bonus)",
    "🎮 5. RL Warning System (Bonus)"
    ])

    # ---- TAB 1: EDA ----
    with vis_tab1:
        st.markdown("<h2 style='text-align: right; color: #38BDF8;'>📊 Environmental Conditions Distribution (750 Accident Samples)</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: right; color: #94A3B8;'>تعداد کل نمونه‌های بررسی‌شده متادیتا: ۱,۵۰۰ نمونه | تعداد نمونه‌های دارای برچسب تصادف (Accident): ۷۵۰ نمونه</p>", unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.subheader("🌦️ Weather Conditions Distribution")
            df_w = pd.DataFrame(list(data['acc_weather'].items()), columns=['Weather', 'Count'])
            df_w['Percentage'] = (df_w['Count'] / 750 * 100).round(1)
            fig_w = px.bar(
                df_w, x='Weather', y='Count', text='Percentage',
                color='Weather', color_discrete_map=COLOR_PALETTE['Weather'],
                title="Accident Count & Percentage by Weather"
            )
            fig_w.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_w, use_container_width=True)

        with col_e2:
            st.subheader("💡 Light Conditions Distribution")
            df_l = pd.DataFrame(list(data['acc_light'].items()), columns=['Light', 'Count'])
            df_l['Percentage'] = (df_l['Count'] / 750 * 100).round(1)
            fig_l = px.pie(
                df_l, names='Light', values='Count',
                color='Light', color_discrete_map=COLOR_PALETTE['Light'], hole=0.4,
                title="Light Condition Share in Accidents"
            )
            fig_l.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_l, use_container_width=True)

        st.subheader("🏙️ Scene Types Distribution")
        df_s = pd.DataFrame(list(data['acc_scene'].items()), columns=['Scene', 'Count'])
        df_s['Percentage'] = (df_s['Count'] / 750 * 100).round(1)
        fig_s = px.bar(
            df_s, x='Count', y='Scene', text='Percentage', orientation='h',
            color='Scene', color_discrete_map=COLOR_PALETTE['Scene'],
            title="Accident Distribution by Scene Type"
        )
        fig_s.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <h4 style="color: #38BDF8; margin-top: 0;">📌 Note on Local Sampling</h4>
            تمام نمونه‌های محلی (Local Sample) که برای آموزش شبکه استفاده شدند، <b>دقیقاً با حفظ همین نسبت‌ها و توزیع کلی متادیتا</b> استخراج شدند تا نماینده واقعی کل دیتاست باشند.
        </div>
        """, unsafe_allow_html=True)

    # ---- TAB 2: BIAS ----
    with vis_tab2:
        st.markdown("<h2 style='text-align: right; color: #F59E0B;'>⚖️ Selection Bias Analysis</h2>", unsafe_allow_html=True)

        st.markdown("""
        <div class="danger-card" style="direction: rtl; text-align: right;">
            <h3 style="margin-top:0;">⚠️ Data Paradox: Is Daylight Urban Driving More Dangerous?</h3>
            <p style="font-size: 1.05rem; line-height: 1.8;">
            اگر به آمار EDA نگاه کنیم احتمالاً فکر می‌کنیم که <b>هوای صاف (۵۹.۳٪)</b>، <b>روز روشن (۹۰.۷٪)</b> و <b>داخل شهر (۶۳.۳٪)</b> خطرناک‌تر است و بیشتر تصادف می‌شود! اما واقعیت این نیست.
            این دقیقاً همان <b>"Data Bias" یا سوگیری داده‌ها (Selection Bias)</b> را نشان می‌دهد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            st.markdown("""
            <div class="glass-card" style="direction: rtl; text-align: right;">
                <h4 style="color: #38BDF8;">🔍 Root Cause of the Bias</h4>
                <p style="font-size: 0.95rem; line-height: 1.7;">
                ماشین‌هایی که این دوربین‌های Nexar را داشتند، بیشتر در ترددهای روزمره شهری و روز روشن در حال حرکت بودند.
                پس طبیعی است که بیشتر تصادفات هم در همین شرایط ضبط شده باشند.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_b2:
            st.markdown("""
            <div class="glass-card" style="direction: rtl; text-align: right;">
                <h4 style="color: #F59E0B;">🎯 Impact on the AI Model</h4>
                <p style="font-size: 0.95rem; line-height: 1.7;">
                مدلی که داریم می‌سازیم قرار است در شرایط عادی شهر و روز خیلی خوب کار کند، ولی چون داده‌ی شب یا هوای بارانی کمتر داشتیم، ممکن است در آن شرایط افت دقت داشته باشد. این یکی از محدودیت‌های ذاتی این دیتاست است که ما از آن آگاهیم و در تحلیل‌هایمان به آن توجه داریم.
                </p>
            </div>
            """, unsafe_allow_html=True)

        bias_df = pd.DataFrame({
            'Condition': ['Day Clear (روز صاف)', 'Day Clear (روز صاف)', 'Night / Rain (شب / باران)', 'Night / Rain (شب / باران)'],
            'Metric': ['Driving Exposure (% تردد کلی)', 'Accidents Recorded (% تصادفات ثبت‌شده)', 'Driving Exposure (% تردد کلی)', 'Accidents Recorded (% تصادفات ثبت‌شده)'],
            'Percentage': [85.0, 90.7, 15.0, 9.3]
        })
        fig_b = px.bar(
            bias_df, x='Condition', y='Percentage', color='Metric', barmode='group',
            color_discrete_map={'Driving Exposure (% تردد کلی)': '#38BDF8', 'Accidents Recorded (% تصادفات ثبت‌شده)': '#EF4444'},
            title="Driving Exposure vs Recorded Accidents"
        )
        fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_b, use_container_width=True)

        # ---- TAB 3: MODEL ITERATIONS ----
    with vis_tab3:
        st.markdown("<h2 style='text-align: right; color: #10B981;'>🧪 Model Iterations (Try 1 to 6)</h2>", unsafe_allow_html=True)

        selected_try = st.selectbox(
            "Select an iteration to explore:",
            [
                "🧪 Try #1: 100 Samples — Severe Overfitting",
                "⚠️ Try #2: Online Processing (400 Samples) — Model Collapse",
                "⚙️ Try #3: Offline ResNet18 Features (1000 Samples)",
                "🏆 Try #4: Champion Model (BiLSTM + Focal Loss + Threshold 0.45)",
                "📉 Try #5: 1400 Samples Without Focal Loss (Regression)",
                "💥 Try #6: Final Model — 1400 Samples, 30 Epochs"
            ]
        )

        st.markdown("---")

        if "Try #1" in selected_try:
            st.markdown("### 🧪 Try #1: Baseline on 100 Samples (Severe Overfitting)")

            col_t1_m, col_t1_i = st.columns([1.2, 1])
            with col_t1_m:
                st.markdown("""
                <div class="warning-card" style="direction: rtl; text-align: right;">
                    <h4>📜 Experiment Scenario & Details</h4>
                    در اولین قدم برای تست اولیه خط لوله، ۱۰۰ نمونه ویدیو وارد شبکه شد.
                    مدل در طی ۱۵ ایپاک به ظاهر عملکرد فوق‌العاده‌ای نشان داد و دقت آموزش از <b>۴۶٪ به ۹۵٪</b> رسید.
                    اما هنگام ارزیابی روی ۲۰ نمونه تست ندیده، دقت به <b>۵۰٪ (در حد شانس محض)</b> سقوط کرد!
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Train Accuracy", "95.0%")
                m2.metric("Test Accuracy", "50.0%")
                m3.metric("Precision", "0.50")
                m4.metric("Recall", "0.50")

            with col_t1_i:
                st.markdown("<b>Confusion Matrix (Try 1):</b>", unsafe_allow_html=True)
                p_output = os.path.join(SCRIPT_DIR, 'output.png')
                p_output2 = os.path.join(SCRIPT_DIR, 'output2.png')
                if os.path.exists(p_output):
                    st.image(p_output, caption="Training Curve & Confusion Matrix — Try #1")
                elif os.path.exists(p_output2):
                    st.image(p_output2, caption="Confusion Matrix — Try #1")
                else:
                    st.info("Confusion Matrix: 50% of accidents and 50% of normal driving were correctly classified.")

        elif "Try #2" in selected_try:
            st.markdown("### ⚠️ Try #2: Online Processing (400 Samples) — Model Collapse")

            st.markdown("""
            <div class="danger-card" style="direction: rtl; text-align: right;">
                <h4>⚡ Online Frame Processing Challenge</h4>
                در این نوبت، دیتاست به ۴۰۰ نمونه ارتقا یافت و تکنیک‌های Data Augmentation (چرخش افقی و Color Jitter) و Optuna اضافه شد.
                با اینکه در آموزش دقت به ۶۷.۵٪ رسید، در دیتای تست مدل دچار <b>Model Collapse (فروپاشی کامل)</b> شد و تمام نمونه‌ها را «عادی» پیش‌بینی کرد!
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", "50.98%", delta="-34.0%")
            c2.metric("Precision", "0.0000", delta="-100%")
            c3.metric("Recall", "0.0000", delta="-100%")
            c4.metric("F1-Score", "0.0000")

            st.markdown("#### 🖼️ Try #2 Visual Outputs (Raw Pixel Bottleneck)")
            img_c1, img_c2, img_c3 = st.columns(3)
            p_400_1 = os.path.join(SCRIPT_DIR, '400sample.png')
            p_400_3 = os.path.join(SCRIPT_DIR, '400sample_3.png')
            p_400_4 = os.path.join(SCRIPT_DIR, '400sample_4.png')
            if os.path.exists(p_400_1):
                img_c1.image(p_400_1, caption="Training Curve — 400 Samples")
            if os.path.exists(p_400_3):
                img_c2.image(p_400_3, caption="Confusion Matrix — Model Collapse")
            if os.path.exists(p_400_4):
                img_c3.image(p_400_4, caption="Threshold Test 0.35 (No Change)")

        elif "Try #3" in selected_try:
            st.markdown("### ⚙️ Try #3: Offline ResNet18 Features (1000 Samples)")

            st.markdown("""
            <div class="safe-card" style="direction: rtl; text-align: right;">
                <h4>💡 Architectural Leap: Offline Vector Extraction</h4>
                برای حل مشکل زمان بسیار بالای پردازش فریم‌های خام و پر شدن رم، هر فریم توسط ResNet18 به یک بردار ۵۱۲-بعدی تبدیل و ذخیره شد.
                سپس ۱۰ آزمایش Optuna با سرعت فوق‌العاده انجام گرفت.
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", "68.50%", delta="+17.5%")
            c2.metric("Precision", "70.33%", delta="+70.3%")
            c3.metric("Recall", "64.00%", delta="+64.0%")
            c4.metric("F1-Score", "67.02%")

            col_m1, col_m2 = st.columns(2)
            p_m1 = os.path.join(SCRIPT_DIR, 'm1.png')
            p_m2 = os.path.join(SCRIPT_DIR, 'm2.png')
            if os.path.exists(p_m1):
                col_m1.image(p_m1, caption="Training & Validation Curve — 1000 Samples")
            if os.path.exists(p_m2):
                col_m2.image(p_m2, caption="Confusion Matrix (Accuracy: 68.5%)")

        elif "Try #4" in selected_try:
            st.markdown("### 🏆 Try #4: Champion Model (BiLSTM + Focal Loss + Threshold 0.45)")

            st.markdown("""
            <div class="glass-card" style="border: 2px solid #10B981; direction: rtl; text-align: right;">
                <h3 style="color: #10B981; margin-top:0;">🌟 Best Performing Architecture (Champion Model)</h3>
                <p style="font-size: 1.05rem; line-height: 1.8;">
                در این مرحله، برای جبران سوگیری داده‌ها از <b>تابع هزینه Focal Loss</b> و شبکه دوطرفه <b>BiLSTM</b> استفاده شد.
                همچنین با جستجوی تعاملی آستانه تصمیم‌گیری (Threshold)، آستانه طلایی <b>0.45</b> انتخاب گردید که به رکورد <b>Recall ۸۳٪</b> (تشخیص ۸۳ تصادف از ۱۰۰ حادثه واقعی) دست یافت.
                </p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", "70.00%", delta="+1.5%")
            c2.metric("Precision", "65.87%")
            c3.metric("Recall", "83.00%", delta="+19.0%")
            c4.metric("F1-Score", "73.45%", delta="+6.4%")

            col_bm1, col_bm2 = st.columns(2)
            p_bm = os.path.join(SCRIPT_DIR, 'bm.png')
            p_bm_matrix = os.path.join(SCRIPT_DIR, 'bm_matrix.png')
            if os.path.exists(p_bm):
                col_bm1.image(p_bm, caption="Training Curve — BiLSTM + Focal Loss (Stopped at Epoch 12)")
            if os.path.exists(p_bm_matrix):
                col_bm2.image(p_bm_matrix, caption="Confusion Matrix — Threshold 0.45 (Recall: 83%)")

            st.markdown("#### 🎛️ Threshold Search: Impact of Decision Threshold")
            thresh_df = pd.DataFrame({
                'Threshold': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
                'Accuracy': [0.6350, 0.7000, 0.7200, 0.6950, 0.6400, 0.5800, 0.5450, 0.5050],
                'Precision': [0.5860, 0.6587, 0.7245, 0.7600, 0.8043, 0.7857, 0.8462, 1.0000],
                'Recall': [0.9200, 0.8300, 0.7100, 0.5700, 0.3700, 0.2200, 0.1100, 0.0100],
                'F1-Score': [0.7160, 0.7345, 0.7172, 0.6514, 0.5068, 0.3438, 0.1947, 0.0198]
            })
            st.dataframe(thresh_df.style.highlight_max(subset=['Recall', 'F1-Score'], color='#065F46'), use_container_width=True)

        elif "Try #5" in selected_try:
            st.markdown("### 📉 Try #5: 1400 Samples Without Focal Loss (Regression)")

            st.markdown("""
            <div class="danger-card" style="direction: rtl; text-align: right;">
                <h4>🧪 Experiment 5: Hypothesis & Outcome</h4>
                برای تست اینکه آیا اضافه کردن ۴۰۰ نمونه داده سخت محیطی (رسیدن به ۱۴۰۰ نمونه) بهتر از Focal Loss جواب می‌دهد، داده‌ها زیاد شدند اما از Focal Loss استفاده نشد.
                نتیجه نشان داد که <b>افزایش کمّی داده به تنهایی نمی‌تواند سوگیری را برطرف کند</b> و مدل مجدداً دچار افت شدید Recall (تنها ۱٪) شد.
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", "50.00%")
            c2.metric("Precision", "50.00%")
            c3.metric("Recall", "1.00%", delta="-82.0%")
            c4.metric("F1-Score", "0.0196")

        else:
            st.markdown("### 💥 Try #6: Final Model — 1400 Samples, 30 Epochs")

            st.markdown("""
            <div class="glass-card" style="direction: rtl; text-align: right;">
                <h4 style="color: #38BDF8;">🚀 Final Training: 1400 Enriched Samples + Early Stopping</h4>
                در این فاز نهایی، شبکه با ۱۴۰۰ نمونه آموزش دید و توقف زودهنگام در ایپاک ۳۰ اعمال شد.
                مدل به توازن بسیار مطلوبی در دیتای تست رسید.
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", "68.00%")
            c2.metric("Precision", "64.29%")
            c3.metric("Recall", "81.00%", delta="+80.0%")
            c4.metric("F1-Score", "71.68%")

            col_f1, col_f2 = st.columns(2)

            epochs_t6 = list(range(1, 31))
            val_acc_t6 = [0.475, 0.475, 0.475, 0.475, 0.475, 0.546, 0.568, 0.539, 0.550, 0.586,
                          0.486, 0.629, 0.714, 0.704, 0.704, 0.750, 0.554, 0.661, 0.732, 0.743,
                          0.671, 0.632, 0.729, 0.793, 0.804, 0.764, 0.807, 0.786, 0.818, 0.761]

            fig_t6 = px.line(
                x=epochs_t6, y=val_acc_t6, markers=True,
                labels={'x': 'Epoch', 'y': 'Validation Accuracy'},
                title="Validation Accuracy (Early Stopping at Epoch 30)"
            )
            fig_t6.update_traces(line_color='#10B981')
            fig_t6.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            col_f1.plotly_chart(fig_t6, use_container_width=True)

            with col_f2:
                st.markdown("<b>Confusion Matrix (Try 6):</b>", unsafe_allow_html=True)
                p_bm_matrix6 = os.path.join(SCRIPT_DIR, 'bm_matrix.png')
                if os.path.exists(p_bm_matrix6):
                    st.image(p_bm_matrix6, caption="Confusion Matrix — 1400 Samples (Accuracy: 68.0%, Recall: 81.0%)")
                else:
                    st.info("81 out of 100 actual accidents were correctly identified by the Phase 6 model.")

        st.markdown("---")
        st.markdown("### 📊 Side-by-Side Comparison of All Iterations")

        all_tries_df = pd.DataFrame([
            {"Iteration": "Try #1 (100 Samples)", "Method": "Raw Frames + CNN", "Test Acc": "50.00%", "Precision": "50.00%", "Recall": "50.00%", "F1-Score": "50.00%", "Status": "Overfitting"},
            {"Iteration": "Try #2 (400 Samples)", "Method": "Online ResNet + LSTM", "Test Acc": "50.98%", "Precision": "0.00%", "Recall": "0.00%", "F1-Score": "0.00%", "Status": "Model Collapse"},
            {"Iteration": "Try #3 (1000 Samples)", "Method": "Offline ResNet + LSTM", "Test Acc": "68.50%", "Precision": "70.33%", "Recall": "64.00%", "F1-Score": "67.02%", "Status": "Good Baseline"},
            {"Iteration": "Try #4 (1000 Samples)", "Method": "BiLSTM + Focal + Thresh 0.45", "Test Acc": "70.00%", "Precision": "65.87%", "Recall": "83.00%", "F1-Score": "73.45%", "Status": "🏆 Champion Model"},
            {"Iteration": "Try #5 (1400 Samples)", "Method": "Enriched Data (No Focal)", "Test Acc": "50.00%", "Precision": "50.00%", "Recall": "1.00%", "F1-Score": "1.96%", "Status": "Recall Collapse"},
            {"Iteration": "Try #6 (1400 Samples)", "Method": "Enriched + Optimized 30 Ep", "Test Acc": "68.00%", "Precision": "64.29%", "Recall": "81.00%", "F1-Score": "71.68%", "Status": "🌟 High Recall Final"}
        ])

        st.dataframe(all_tries_df, use_container_width=True)

        # ---- TAB 4: ADAS Real-Time Demo ----
    with vis_tab4:
        st.markdown("<h2 style='text-align: right; color: #C084FC;'>🚀 ADAS: Real-Time Streaming & Autonomous Decision System (Bonus)</h2>", unsafe_allow_html=True)

        # --- Story 1: Why Unidirectional LSTM ---
        st.markdown("""
        <div class="bonus-card" style="direction: rtl; text-align: right;">
            <h3 style="color: #F3E8FF; margin-top: 0;">⚡ Why Unidirectional LSTM for Live Streaming?</h3>
            <p style="font-size: 1.05rem; line-height: 1.8;">
            اگرچه شبکه دوطرفه <b>BiLSTM</b> در تحلیل آفلاین عملکرد بالایی داشت، اما در یک <b>سیستم استریم زنده (Live Streaming)</b> قابل استفاده نیست!
            زیرا BiLSTM برای پیش‌بینی نیاز به مشاهده فریم‌های آینده دارد، در حالی که در رانندگی واقعی فریم‌های آینده هنوز اتفاق نیفتاده‌اند.
            <br><br>
            بنابراین برای پیاده‌سازی مدل تجاری واقعی از شبکه <b>یک‌طرفه (Unidirectional LSTM)</b> استفاده کردیم و پس از آن با <b>Optuna</b> بهترین ابرپارامترها را جست‌وجو کردیم.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Story 2: Two-stream fusion ---
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border-right: 4px solid #38BDF8;">
            <h3 style="color: #38BDF8; margin-top: 0;">🧠 Two-Stream Sensor Fusion: What + When</h3>
            <p style="font-size: 1.0rem; line-height: 1.8;">
            تصمیم نهایی بر اساس ترکیب دو جریان اطلاعاتی گرفته می‌شود:
            <br>• <b>LSTM (When)</b> → احتمال زمانی وقوع تصادف را از روی توالی فریم‌ها محاسبه می‌کند.
            <br>• <b>YOLOv8n (What)</b> → موقعیت و فاصله‌ی خودروهای اطراف را در لحظه شناسایی می‌کند.
            <br><br>
            سپس با یک <b>Fusion Mask</b> بین motion mask (اختلاف فریم) و vehicle mask (خروجی YOLO)، منطقه‌ی خطر در سه ناحیه‌ی چپ/مرکز/راست بررسی می‌شود و فرمان مناسب صادر می‌گردد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Champion metrics ---
        st.markdown("### 🏆 Real-Time Champion Model Performance (Threshold = 0.45)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", "68.00%", delta="-2.00% vs BiLSTM")
        c2.metric("Precision", "62.86%", delta="-3.01%")
        c3.metric("Recall (Safety)", "88.00%", delta="+5.00% 🚀")
        c4.metric("F1-Score", "73.33%", delta="≈ same")

        st.markdown("""
        <div class="warning-card" style="direction: rtl; text-align: right;">
            <h4>📌 Trade-off Design Note</h4>
            نسخه‌ی Real-Time نسبت به BiLSTM آفلاین کمی Precision از دست داد، اما <b>Recall از ۸۳٪ به ۸۸٪</b> رسید.
            در یک سیستم ایمنی خودرو، <b>از دست دادن یک تصادف بسیار خطرناک‌تر از یک هشدار اضافی</b> است؛ بنابراین این Trade-off کاملاً هدفمند است.
        </div>
        """, unsafe_allow_html=True)

        # --- Optuna hyperparams ---
        st.markdown("### 🎯 Optuna-Tuned Hyperparameters (30 trials)")
        st.code(
            "hidden_size=256  |  num_layers=1  |  dropout=0.52  |  lr=6.2e-4  |  weight_decay=6.5e-4",
            language="text"
        )

        # --- Images ---
        st.markdown("### 📈 Training & Evaluation")
        col_i1, col_i2 = st.columns(2)
        img1_path = os.path.join(SCRIPT_DIR, 'emtiazi1_1.png')
        img2_path = os.path.join(SCRIPT_DIR, 'emtiazi1_2.png')
        if os.path.exists(img1_path):
            col_i1.image(img1_path, caption="One-Way LSTM — Training & Validation Curve")
        if os.path.exists(img2_path):
            col_i2.image(img2_path, caption="Confusion Matrix @ Threshold 0.45 (Recall = 88%)")

        # --- Threshold table ---
        st.markdown("### 🎛️ Threshold Search — Decision Boundary Sweep")
        thresh_a = pd.DataFrame({
            'Threshold': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
            'Accuracy':  [0.6200, 0.6800, 0.6900, 0.6950, 0.6750, 0.5850, 0.5100, 0.5000],
            'Precision': [0.5723, 0.6286, 0.6638, 0.7097, 0.8070, 1.0000, 1.0000, 0.0000],
            'Recall':    [0.9500, 0.8800, 0.7700, 0.6600, 0.4600, 0.1700, 0.0200, 0.0000],
            'F1-Score':  [0.7143, 0.7333, 0.7130, 0.6839, 0.5860, 0.2906, 0.0392, 0.0000],
        })
        st.dataframe(thresh_a.style.highlight_max(subset=['F1-Score'], color='#065F46'), use_container_width=True)

                # --- Videos ---
        st.markdown("---")
        st.markdown("### 🎬 Live ADAS Demo Videos")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### 📹 Accident Scenario Demo")
            p1 = os.path.join(SCRIPT_DIR, 'emtiazi1_video1.mp4')
            if os.path.exists(p1):
                st.video(p1, format="video/mp4")
            else:
                st.warning("⚠️ فایل `emtiazi1_video1.mp4` در کنار اسکریپت پیدا نشد.")

        with col_v2:
            st.markdown("#### 📹 Normal Driving Demo")
            p2 = os.path.join(SCRIPT_DIR, 'emtiazi1_video2.mp4')
            if os.path.exists(p2):
                st.video(p2, format="video/mp4")
            else:
                st.warning("⚠️ فایل `emtiazi1_video2.mp4` در کنار اسکریپت پیدا نشد.")

        # --- Decision gauge simulator ---
        st.markdown("---")
        st.markdown("### 🎛️ Live Autonomous Control Gauge Simulator")

        sim_risk = st.slider("Simulate Frame Risk Probability (%):", 0, 100, 75)

        col_cmd1, col_cmd2, col_cmd3 = st.columns(3)

        if sim_risk < 35:
            col_cmd1.markdown("""
            <div class="safe-card" style="text-align: center;">
                <h2>🟢 NORMAL CRUISE</h2>
                <p>وضعیت عادی | حفظ سرعت و مسیر</p>
            </div>
            """, unsafe_allow_html=True)
            col_cmd2.metric("Driver Warning Level", "SAFE")
            col_cmd3.metric("Autonomous Command", "NO ACTION")

        elif sim_risk < 70:
            col_cmd1.markdown("""
            <div class="warning-card" style="text-align: center;">
                <h2>⚠️ CAUTION ALERT</h2>
                <p>ریسک متوسط | هشدار صوتی به راننده</p>
            </div>
            """, unsafe_allow_html=True)
            col_cmd2.metric("Driver Warning Level", "AUDIBLE WARNING", delta="HIGH RISK")
            col_cmd3.metric("Autonomous Command", "PREPARE BRAKES")

        else:
            col_cmd1.markdown("""
            <div class="danger-card" style="text-align: center;">
                <h2>🚨 CRITICAL COLLISION</h2>
                <p>ریسک بحرانی برخورد | واکنش ترمز اضطراری و انحراف</p>
            </div>
            """, unsafe_allow_html=True)
            col_cmd2.metric("Driver Warning Level", "EMERGENCY ALARM", delta_color="inverse", delta="CRITICAL")
            col_cmd3.metric("Autonomous Command", "🛑 EMERGENCY BRAKE & ↙️ STEER LEFT")
            
    # ---- TAB 5: RL Warning System ----
    with vis_tab5:
        st.markdown("<h2 style='text-align: right; color: #C084FC;'>🎮 Reinforcement Learning Warning Decision System (Bonus)</h2>", unsafe_allow_html=True)

        # --- Story: The Question ---
        st.markdown("""
        <div class="bonus-card" style="direction: rtl; text-align: right;">
            <h3 style="color: #F3E8FF; margin-top: 0;">❓ The Core Question: When Should the System Alert?</h3>
            <p style="font-size: 1.05rem; line-height: 1.8;">
            LSTM به ما می‌گوید «احتمال تصادف در این لحظه چقدر است؟»، اما یک سؤال کلیدی باقی می‌ماند:
            <b>چه زمانی باید به راننده هشدار داد؟</b> اگر خیلی زود هشدار دهیم، هشدارهای کاذب باعث بی‌اعتنایی راننده می‌شوند. اگر دیر هشدار دهیم، فرصت واکنش از بین می‌رود.
            <br><br>
            این یک مسئله‌ی <b>تصمیم‌گیری متوالی</b> است — نه یک threshold ساده. بنابراین از <b>Reinforcement Learning (DQN)</b> استفاده کردیم تا Agent با تجربه‌ی هزاران ویدئو یاد بگیرد چه زمانی هشدار بدهد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Story: Reward design ---
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border-right: 4px solid #F59E0B;">
            <h3 style="color: #F59E0B; margin-top: 0;">🎯 Reward Design — Why These Numbers?</h3>
            <p style="font-size: 0.98rem; line-height: 1.9;">
            • هشدار در پنجره‌ی <b>0 تا 5 ثانیه</b> قبل از تصادف → <b>+15</b> (زمان ایده‌آل واکنش)<br>
            • هشدار بیش از 5 ثانیه زودتر → <b>+3</b> (مفید اما زودهنگام)<br>
            • هشدار روی ویدئوی عادی → <b>-5</b> (هشدار کاذب — راننده را بی‌حوصله می‌کند)<br>
            • <b>عدم هشدار</b> در ویدئوی تصادف → <b>-15</b> (بدترین سناریو — تصادف از دست رفته)<br>
            • پایان صحیح ویدئوی عادی بدون هشدار → <b>+5</b><br>
            • هر فریم انتظار → <b>-0.005</b> (پاداش منفی خیلی کوچک برای تشویق تصمیم‌گیری سریع)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Story: Three versions ---
        st.markdown("### 📖 The Evolution: v1 → v3 → v4")

        col_v1_story, col_v3_story, col_v4_story = st.columns(3)

        with col_v1_story:
            st.markdown("""
            <div class="danger-card" style="direction: rtl; text-align: right; height: 100%;">
                <h4 style="margin-top: 0;">❌ v1 — State of 4 (Failed)</h4>
                <p style="font-size: 0.9rem; line-height: 1.7;">
                <b>State:</b> [risk, delta, max_risk, elapsed] — عملاً فقط risk.<br><br>
                <b>مشکل بنیادی:</b> وقتی ورودی فقط یک عدد باشد، policy بهینه دقیقاً یک threshold است. DQN داشت تلاش می‌کرد یک threshold یاد بگیرد — در حالی که grid-search سریع‌تر و دقیق‌تر عمل می‌کرد.
                </p>
                <p style="color: #FCA5A5; font-weight: bold; margin-bottom: 0;">Reward = -2.17</p>
            </div>
            """, unsafe_allow_html=True)

        with col_v3_story:
            st.markdown("""
            <div class="warning-card" style="direction: rtl; text-align: right; height: 100%;">
                <h4 style="margin-top: 0;">🔶 v3 — State of 10 + YOLO</h4>
                <p style="font-size: 0.9rem; line-height: 1.7;">
                <b>State:</b> risk + YOLO features (n_veh, min_d_c, min_d_l, min_d_r, n_close).<br><br>
                <b>ایده:</b> حالا Agent دو شاهد دارد — LSTM می‌گوید «کی خطرناک است»، YOLO می‌گوید «صحنه چطور است».
                </p>
                <p style="color: #FCD34D; font-weight: bold; margin-bottom: 0;">Reward = -0.52 (بهبود +1.65)</p>
            </div>
            """, unsafe_allow_html=True)

        with col_v4_story:
            st.markdown("""
            <div class="safe-card" style="direction: rtl; text-align: right; height: 100%; border: 2px solid #10B981;">
                <h4 style="margin-top: 0;">🏆 v4 — State of 11 + VecNormalize (Winner)</h4>
                <p style="font-size: 0.9rem; line-height: 1.7;">
                <b>سه تغییر کلیدی:</b><br>
                ① VecNormalize (یکسان‌سازی مقیاس feature‌ها)<br>
                ② Val split واقعی (۱۵۰ ویدئو)<br>
                ③ feature جدید risk_rise (چند فریم از وقتی risk > 0.4 بود)
                </p>
                <p style="color: #6EE7B7; font-weight: bold; margin-bottom: 0;">Reward = +0.36 ✓ بالاترین</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Insight badge ---
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border: 2px solid #C084FC;">
            <h3 style="color: #C084FC; margin-top: 0;">💡 The Scientific Insight We Discovered</h3>
            <p style="font-size: 1.05rem; line-height: 1.9;">
            <b>کیفیت State مهم‌تر از انتخاب الگوریتم است!</b><br>
            وقتی state فقط risk بود (v1)، حتی DQN هم نتوانست از threshold ساده‌تر پیشی بگیرد. اما وقتی state شامل اطلاعات مکانی YOLO و روند زمانی risk شد (v4)، DQN به‌طور واقعی از همه‌ی threshold‌ها پیشی گرفت — حتی از بهترین threshold (th=0.50 با +0.35).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- Results table ---
        st.markdown("### 📊 Full Comparison: DQN Versions vs Thresholds")

        rl_results = pd.DataFrame([
            {'Method': 'DQN v1 (state 4)',      'Detection': 0.37, 'False Alarm': 0.10, 'Lead (s)': 9.68, 'Reward': -2.17, 'Verdict': '❌ Fails'},
            {'Method': 'DQN v3 (state 10)',     'Detection': 0.71, 'False Alarm': 0.37, 'Lead (s)': 9.93, 'Reward': -0.52, 'Verdict': '🔶 Better'},
            {'Method': 'DQN v4 (state 11) 🏆',  'Detection': 0.81, 'False Alarm': 0.53, 'Lead (s)': 7.98, 'Reward':  0.36, 'Verdict': '✅ Winner'},
            {'Method': 'Threshold 0.45',        'Detection': 0.97, 'False Alarm': 0.77, 'Lead (s)': 10.01,'Reward':  0.06, 'Verdict': 'Aggressive'},
            {'Method': 'Threshold 0.50',        'Detection': 0.88, 'False Alarm': 0.55, 'Lead (s)': 9.88, 'Reward':  0.35, 'Verdict': 'Best Threshold'},
            {'Method': 'Threshold 0.55',        'Detection': 0.76, 'False Alarm': 0.37, 'Lead (s)': 9.74, 'Reward':  0.23, 'Verdict': 'Conservative'},
            {'Method': 'Threshold 0.60',        'Detection': 0.55, 'False Alarm': 0.20, 'Lead (s)': 9.29, 'Reward': -0.87, 'Verdict': 'Too Late'},
        ])
        st.dataframe(rl_results, use_container_width=True)

        # --- Images ---
        st.markdown("### 📈 Training Curves & Final Comparison")
        col_i1, col_i2 = st.columns(2)
        img1_path = os.path.join(SCRIPT_DIR, 'emtiazi2_1.png')
        img2_path = os.path.join(SCRIPT_DIR, 'emtiazi2_2.png')
        if os.path.exists(img1_path):
            col_i1.image(img1_path, caption="Risk over Time with Event Marker (sample accident video)")
        if os.path.exists(img2_path):
            col_i2.image(img2_path, caption="Detection vs False Alarm Trade-off + Reward Comparison")

                # --- Video ---
        st.markdown("---")
        st.markdown("### 🎬 RL-Based ADAS Demo Video")

        rl_vid_path = os.path.join(SCRIPT_DIR, 'emtiazi2_video1.mp4')
        if os.path.exists(rl_vid_path):
            st.video(rl_vid_path, format="video/mp4")
        else:
            st.warning("⚠️ فایل `emtiazi2_video1.mp4` در کنار اسکریپت پیدا نشد.")
# ==========================================
# PAGE 2: NLP
# ==========================================
else:
    # ---------- helpers ----------
    def show_nlp_pair(f1, f2, c1, c2):
        col1, col2 = st.columns(2)
        for col, fn, cap in [(col1, f1, c1), (col2, f2, c2)]:
            p = os.path.join(SCRIPT_DIR, fn)
            if os.path.exists(p):
                col.image(p, caption=cap)
            else:
                col.warning(f"⚠️ تصویر پیدا نشد: `{fn}`")

    def show_nlp_single(fn, cap):
        p = os.path.join(SCRIPT_DIR, fn)
        if os.path.exists(p):
            st.image(p, caption=cap)
        else:
            st.warning(f"⚠️ تصویر پیدا نشد: `{fn}`")

    # ---------- header ----------
    st.markdown("""
    <div class="hero-card" style="direction: rtl; text-align: right;">
        <h1 style="color: #38BDF8; margin: 0; font-weight: 800;">💬 NLP & Customer Sentiment Analysis</h1>
        <p style="color: #CBD5E1; font-size: 1.1rem; margin-top: 8px;">
        تحلیل متنی نظرات خریداران، ارزیابی رضایت از گارانتی، مدل‌سازی احساس با RoBERTa و توالی با HMM
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 5 tabs ----------
    nlp_tab1, nlp_tab2, nlp_tab3, nlp_tab4 = st.tabs([
        "📊 1. EDA",
        "⭐ 2. Warranty Satisfaction",
        "🤖 3. Sentiment Model",
        "🔗 4. HMM Sequence Modeling"
    ])

       # ==========================================
    # TAB 1: EDA — Exploratory Data Analysis (6 Questions)
    # ==========================================
    with nlp_tab1:
        st.markdown("<h2 style='text-align: right; color: #38BDF8;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

        st.markdown("""
        <div class="hero-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1.02rem; line-height: 1.9;">
            وقتی پروژه را شروع کردیم، قبل از اینکه سراغ ساخت مدل برویم، باید به یک سؤال اساسی جواب می‌دادیم:
            داده‌های ما چه ویژگی‌هایی دارند و چه چیزهایی ممکن است مدل را به اشتباه بیندازد؟
            برای همین، تحلیل اکتشافی را در شش بخش انجام دادیم؛
            از توزیع امتیازها تا محتوای نظرها، رفتار کاربران و مقایسه‌ی محصولات و برندها.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Step 0: Data Overview & Product Table Join")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            داده‌ی اصلی ما شامل <b>۸۳۸ هزار و ۹۴۴ نظر</b> و <b>۱۱ ستون</b> بود.
            برای اضافه‌کردن عنوان و برند محصولات، جدول اطلاعات محصول را با شناسه‌ی <code>asin</code> به نظرات متصل کردیم.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            یک نکته‌ی مهم پیدا کردیم: جدول محصول، <b>۳۰ هزار و ۳۶۸ ردیف با شناسه‌ی تکراری</b> داشت و اتصال مستقیم،
            تعداد نظرات را به‌اشتباه افزایش می‌داد. بنابراین برای هر شناسه، اولین ردیف را نگه داشتیم
            و بررسی کردیم که بعد از اتصال، تعداد نظرات همان مقدار اولیه باقی بماند.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### ❓ Question 1: آیا امتیازهای ۱ تا ۵ توزیع متوازنی دارند؟")
        st.markdown("""
        <div class="danger-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            نمودار نشان داد که <b>نه</b>؛ بیش از <b>۴۶۱ هزار نظر</b> پنج‌ستاره بودند، یعنی حدود <b>۵۵ درصد</b> کل داده‌ها.
            در مقابل، امتیاز دو فقط حدود <b>۵۷ هزار نمونه</b> داشت.
            بنابراین مدل از ابتدا با تعداد بسیار بیشتری نمونه‌ی پنج‌ستاره روبه‌رو می‌شد.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            در این بخش، نامتوازن‌بودن داده‌ها را شناسایی کردیم؛ <b>خودِ متوازن‌سازی در این نوت‌بوک اجرا نشده است.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_single('eda1.png', 'Q1: Rating Distribution — Highly Imbalanced Dataset')

        st.markdown("---")

        st.markdown("### ❓ Question 2: کاربران در نظرات مثبت، خنثی و منفی بیشتر درباره‌ی چه چیزهایی صحبت می‌کنند؟")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            امتیازهای ۴ و ۵ را مثبت، ۳ را خنثی و ۱ و ۲ را منفی در نظر گرفتیم.
            از هر دسته، <b>۲۰ هزار نظر تصادفی</b> برداشتیم و ابر کلمات ساختیم.
            علاوه بر کلمات زائد استاندارد، کلمات عمومی مثل <code>product</code> و <code>amazon</code> را هم کنار گذاشتیم؛
            ولی کلمات نفی مثل <code>not</code> و <code>no</code> را حفظ کردیم.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            در نظرات مثبت، کلماتی مثل <b>great</b>، <b>good</b>، <b>love</b> و <b>easy</b> برجسته بودند.
            در منفی‌ها، کلماتی مثل <b>problem</b>، <b>return</b>، <b>tried</b> و <b>work</b> دیده می‌شدند.
            دسته‌ی خنثی هم ترکیبی از واژه‌های مثبت و منفی داشت.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_single('eda2.png', 'Q2: Word Clouds — Positive / Neutral / Negative Reviews')

        st.markdown("""
        <div class="warning-card" style="direction: rtl; text-align: right;">
            <h4 style="margin-top: 0;">💡 Key Insight: Word Overlap</h4>
            <p style="font-size: 1rem; line-height: 1.9;">
            نکته‌ی جالب این بود که بین <b>۵۰ واژه‌ی اول مثبت و منفی، ۳۲ واژه مشترک</b> بود.
            پس صرفِ دیدن یک کلمه برای تشخیص احساس کافی نیست؛ مثلاً <code>good</code> در ترکیب <code>not good</code>
            معنای متفاوتی پیدا می‌کند. این نتیجه، اهمیت حفظ کلمات نفی و توجه به بافت جمله را برای ادامه‌ی پروژه روشن کرد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### ❓ Question 3: کاربرانی که نظراتشان بیشترین رأی مفیدبودن را گرفته‌اند")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            ستون <code>vote</code> را عددی کردیم، کاماها را برداشتیم و مقادیر خالی یا نامعتبر را صفر گذاشتیم.
            بعد مجموع رأی‌ها را بر اساس شناسه‌ی هر کاربر حساب کردیم تا تفاوت اسم‌ها باعث تقسیم‌شدن رأی‌های یک نفر نشود.
            </p>
        </div>
        """, unsafe_allow_html=True)

        top_reviewers_df = pd.DataFrame({
            'Rank': ['#1', '#2', '#3', '...', '#10'],
            'Reviewer': ['Benson Leung', 'Voltron00x', 'Obi Wan', '...', 'Mark A. Baker'],
            'Total Votes': [4572, 3181, 3150, '...', 1820],
            'Note': ['روی ۲۹ نظر', '—', '—', '—', 'فقط یک نظر']
        })
        st.dataframe(top_reviewers_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warning-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            <b>📌 نکته:</b> نفر دهم، <b>Mark A. Baker</b>، با <b>فقط یک نظر</b> ۱۸۲۰ رأی گرفته بود؛
            پس <b>تعداد نظر بیشتر، لزوماً به معنی رأی مفیدبودن بیشتر نیست.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### ❓ Question 4: طول متن نظرات بر حسب تعداد کاراکتر")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            دو هیستوگرام با <b>۱۰۰ بازه</b> رسم کردیم؛ یکی برای تمام متن‌ها و یکی برای فاصله‌ی صدک اول تا نودونهم،
            یعنی <b>۲۰۲ تا ۳۳۹۹ کاراکتر</b>. توزیع به‌شدت راست‌چوله بود:
            میانه‌ی طول متن <b>۴۰۸ کاراکتر</b> بود و <b>۹۰ درصد</b> نظرها حداکثر <b>۱۲۱۷ کاراکتر</b> داشتند.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            فیلترکردن، شکل بخش اصلی توزیع را خواناتر کرد. پیشنهاد ثبت‌شده برای مدل‌سازی، بررسی همین محدوده
            یا کوتاه‌کردن ورودی ترنسفورمر در حدود <b>۲۵۶ تا ۵۱۲ توکن</b> بود؛ اینجا توکن با کاراکتر تفاوت دارد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_single('eda3.png', 'Q4: Review Length Distribution — Raw vs Filtered (P1–P99)')

        st.markdown("---")

        st.markdown("### ❓ Question 5: محصولاتی که بیشترین تعداد نظر ۵ ستاره را دارند")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            فقط نظرهای پنج‌ستاره را انتخاب و بر اساس شناسه‌ی محصول شمارش کردیم.
            </p>
        </div>
        """, unsafe_allow_html=True)

        top_products_5star = pd.DataFrame({
            'Rank': ['#1', '#2', '#3'],
            'Product': ['OontZ Angle 3 Speaker', 'NETGEAR WiFi Amplifier', 'Roku Streaming Stick'],
            'Brand': ['Cambridge Soundworks', 'NETGEAR', 'Roku'],
            '5-Star Reviews': [1163, 659, 600]
        })
        st.dataframe(top_products_5star, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warning-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            <b>📌 نکته مهم:</b> این رتبه‌بندی، <b>تعداد</b> نظر پنج‌ستاره را نشان می‌دهد؛
            برای نتیجه‌گیری درباره‌ی <b>نسبت رضایت</b>، به تعداد کل نظرات هر محصول هم نیاز داریم.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### ❓ Question 6: مقایسه‌ی برندها — تعداد نظر و میانگین امتیاز")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            ابتدا ده برند با بیشترین تعداد نظر را انتخاب کردیم و بعد همان ده برند را بر اساس میانگین امتیاز مرتب کردیم.
            <b>۱۲۳۱ نظر بدون برند</b> (حدود ۰.۱۵ درصد داده‌ها) وارد این مقایسه نشدند.
            </p>
        </div>
        """, unsafe_allow_html=True)

        brand_df = pd.DataFrame({
            'Brand': ['Logitech', 'Anker', 'AmazonBasics', 'NETGEAR'],
            'Review Count': [16993, '—', '—', '—'],
            'Avg Rating': ['—', 4.244, 4.201, 3.810],
            'Note': ['بیشترین تعداد نظر', 'بالاترین میانگین امتیاز', 'دوم', 'پایین‌ترین میانگین بین ۱۰ برند']
        })
        st.dataframe(brand_df, use_container_width=True, hide_index=True)

        show_nlp_single('eda4.png', 'Q6: Top 10 Brands — Review Count vs Average Rating')

        st.markdown("---")

        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border: 2px solid #38BDF8;">
            <h4 style="color: #38BDF8; margin-top: 0;">📌 Takeaways from EDA</h4>
            <p style="font-size: 1rem; line-height: 1.9;">
            این تحلیل به ما نشان داد که برای ادامه‌ی پروژه باید:
            </p>
            <ul style="font-size: 0.98rem; line-height: 1.9;">
                <li>عدم توازن امتیازها را جدی بگیریم.</li>
                <li>معنای جمله و کلمات نفی را حفظ کنیم.</li>
                <li>طول ورودی‌ها را مدیریت کنیم.</li>
                <li>موقع تفسیر رتبه‌بندی‌ها، تفاوت تعداد نظر و میانگین امتیاز را در نظر داشته باشیم.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # TAB 2: WARRANTY SATISFACTION
    # ==========================================
    with nlp_tab2:
        st.markdown("<h2 style='text-align: right; color: #10B981;'>⭐ Warranty Satisfaction Analysis</h2>", unsafe_allow_html=True)

        # --- Section 1: Average Rating Comparison (n1) ---
        st.markdown("### 📊 Chart 1: Average Rating — All Reviews vs Warranty Reviews")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            نتایج تحلیل میانگین امتیازها نشان می‌دهد که رضایت کلی کاربران از محصولات آمازون در سطح بالاتری نسبت به تجربه آن‌ها از خدمات گارانتی قرار دارد.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            میانگین امتیاز تمامی نظرات برابر با <b>۴.۰۲ از ۵</b> است؛ درحالی‌که میانگین امتیاز نظرات مرتبط با گارانتی تنها <b>۳.۲۰ از ۵</b> را نشان می‌دهد.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            این اختلاف <b>۰.۸۲ امتیازی</b> بیانگر آن است که تجربه مشتریان از گارانتی به اندازه تجربه کلی آن‌ها از محصولات رضایت‌بخش نیست.
            به عبارت دیگر، ممکن است کاربران از کیفیت و عملکرد محصول رضایت داشته باشند، اما در زمینه خدمات گارانتی و پشتیبانی با تجربه ضعیف‌تری مواجه شوند.
            </p>
        </div>
        """, unsafe_allow_html=True)
        show_nlp_single('n1.png', 'Chart 1: Average Rating Comparison — Key Insight')


        # --- Section 3: Rating Distribution (n2) ---
        st.markdown("### ⭐ Chart 2: Rating Distribution in Warranty Reviews")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            بررسی توزیع امتیاز نظرات مرتبط با گارانتی نشان می‌دهد که بیشترین فراوانی مربوط به امتیاز ۵ با سهم <b>۳۷.۲ درصد</b> است که بیانگر تجربه مثبت بخش قابل‌توجهی از کاربران است.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            در مقابل، امتیاز ۱ با سهم <b>۲۸.۳ درصد</b> در جایگاه دوم قرار دارد و نشان‌دهنده وجود نارضایتی شدید در میان گروهی از مشتریان است.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            نکته قابل‌توجه، فاصله نسبتاً زیاد میان امتیازهای ابتدا و انتهای طیف و سهم کمتر امتیازهای میانی است.
            این الگو می‌تواند نشان‌دهنده تفاوت چشمگیر تجربه کاربران در مواجهه با خدمات گارانتی باشد؛
            به‌طوری‌که برخی کاربران تجربه‌ای کاملاً رضایت‌بخش و برخی دیگر تجربه‌ای نامطلوب داشته‌اند.
            </p>
        </div>
        """, unsafe_allow_html=True)
        show_nlp_single('n2.png', 'Chart 2: Rating Distribution — Key Insight')

        st.markdown("---")

        # --- Section 4: Scatter Dispersion (n3) ---
        st.markdown("### 📉 Chart 3: Dispersion vs Review Volume")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            این نمودار پراکندگی، ارتباط میان تعداد نظرات ثبت‌شده برای محصولات و میانگین امتیاز نظرات مرتبط با گارانتی را نمایش می‌دهد.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            بررسی پراکندگی داده‌ها نشان می‌دهد که امتیازهای گارانتی در تعداد نظرات مختلف، دامنه گسترده‌ای از ۱ تا ۵ را پوشش می‌دهند.
            به‌عبارت دیگر، حتی محصولاتی با تعداد نظرات مشابه نیز می‌توانند تجربه‌های متفاوتی از نظر رضایت مشتریان داشته باشند.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            همچنین با افزایش تعداد نظرات، نقاط همچنان در سطوح مختلف امتیازی پراکنده هستند و روند صعودی یا نزولی مشخصی مشاهده نمی‌شود.
            این موضوع نشان می‌دهد که تعداد نظرات به‌تنهایی شاخص مناسبی برای پیش‌بینی رضایت مشتریان از گارانتی نیست.
            </p>
        </div>
        """, unsafe_allow_html=True)
        show_nlp_single('n3.png', 'Chart 3: Dispersion — Key Insight')

        # ==========================================
    # TAB 3: SENTIMENT MODEL — RoBERTa-large
    # ==========================================
    with nlp_tab3:
        st.markdown("<h2 style='text-align: right; color: #38BDF8;'>🤖 Sentiment Analysis — RoBERTa-large</h2>", unsafe_allow_html=True)

        st.markdown("### 🎯 Goal & Problem Statement")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            در فاز دوم پروژه، تمرکز روی پیش‌بینی پیوسته و ترتیبی امتیاز نظرات آمازون (۱ تا ۵ ستاره) از روی متن قرار گرفت.
            داده‌های اولیه شامل <b>۸۳۸,۹۴۴ رکورد آموزشی</b> و کاتالوگ محصولات با بیش از ۷۵۰ هزار سطر یکتا بود.
            چالش اصلی، جداسازی ظریف احساسات در کلاس‌های میانی (۲، ۳ و ۴) و کنترل عدم تعادل شدید میان کلاس‌ها بود.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🛡️ Data Preparation & Leakage Guard")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            برای اطمینان از سلامت سنجش، ابتدا یک مجموعه ارزیابی (Validation) کاملاً متوازن شامل <b>۳۰,۰۰۰ سطر</b>
            (دقیقاً ۶,۰۰۰ نمونه برای هر کدام از ۵ کلاس) جدا شد.
            سپس <b>۲,۵۲۸ نمونه هم‌پوشان</b> با مجموعه ارزیابی و <b>۲۷,۸۷۲ نقد تکراری</b> از داده‌های آموزش حذف شدند
            تا یک استخر آموزشی ایزوله با <b>۷۷۸,۵۴۴ سطر خالص</b> تشکیل شود.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🧹 Minimal Cleaning Preprocessing")
        st.markdown("""
        <div class="safe-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            به جای پیش‌پردازش‌های سنتی که لحن و شدت احساسات را از بین می‌برند، رویکرد <b>پاک‌سازی حداقلی</b> اعمال شد:
            </p>
            <ul style="font-size: 0.98rem; line-height: 1.9;">
                <li>پاک‌سازی صرفاً به حذف تگ‌های HTML، آدرس‌های وب (URLs) و فاصله‌های مازاد محدود شد.</li>
                <li>حروف بزرگ، علائم نگارشی و ساختار جملات حفظ شدند تا شدت هیجان و خشم در نقد باقی بماند.</li>
                <li>ایموجی‌ها حذف نشدند و به صورت کاراکترهای خام یونیکد (Raw Unicode) به توکنایزر <b>Byte-Level BPE</b> داده شدند تا در سطح بایت به عنوان سیگنال احساسی عمل کنند.</li>
                <li>ورودی مدل با قالب استاندارد <code>Summary: {summary} | Review: {review}</code> ترکیب شد و نام برند برای جلوگیری از اضافه شدن نویز آماری کنار گذاشته شد.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🔄 Rotating Balanced Sampler")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            به دلیل عدم تعادل شدید در داده‌ها (کلاس ۵ دارای بیش از ۴۳۸ هزار و کلاس ۲ دارای حدود ۴۸ هزار سطر)،
            استفاده از روش‌های مرسوم حذف داده باعث از دست رفتن تنوع و کاهش تعمیم‌پذیری می‌شد.
            با پیاده‌سازی نمونه‌بردار چرخشی، در هر اپوک <b>۱۰,۰۰۰ نمونه متوازن از هر کلاس</b>
            (مجموعاً ۵۰,۰۰۰ داده در هر اپوک) به مدل تزریق شد.
            به این ترتیب بدون سربار حافظه، مدل در طول ۶ اپوک طیف متنوعی از داده‌های هر کلاس را دید.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 🏗️ RoBERTa-large & Custom Loss")
        st.markdown("""
        <div class="bonus-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            با بررسی مدل‌های پایه نظیر <b>DistilBERT</b> (دقت ۶۵٪) و <b>DeBERTa-base</b> (دقت ۶۷٪)،
            مدل <b>RoBERTa-large</b> انتخاب شد. معماری و استراتژی آموزش:
            </p>
            <ul style="font-size: 0.98rem; line-height: 1.9;">
                <li><b>استخراج ویژگی:</b> استفاده از مکانیزم Masked Mean Pooling روی آخرین لایه ترنسفورمر، به همراه LayerNorm و Dropout برابر با ۰.۱.</li>
                <li><b>تابع زیان ترتیبی (Custom Loss):</b> به دلیل ترتیبی بودن نمرات، تابعی ترکیبی با وزن‌های مشخص به کار رفت:
                    <br><code>Loss = 0.70 × CrossEntropy (Label Smoothing 0.1) + 0.15 × Ordinal BCE + 0.15 × Smooth L1</code>
                    <br>این ترکیب تضمین کرد که خطاهای فاصله دور (مثل پیش‌بینی نمره ۱ به جای ۵) به مراتب جریمه سنگین‌تری نسبت به خطاهای همسایه دریافت کنند.
                </li>
                <li><b>تنظیمات آموزش:</b> نرخ یادگیری با روش Layer-wise LR Decay (ضریب ۰.۹۵)، نرخ پایه ۲e-۵ برای ستون فقرات مدل، و اندازه دسته موثر ۳۲ با Gradient Accumulation.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 📊 Results & Confusion Matrix Analysis")
        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.9;">
            مدل به دقت کلی <b>۷۰٪</b> روی داده‌های ارزیابی متوازن دست یافت:
            </p>
            <ul style="font-size: 0.98rem; line-height: 1.9;">
                <li>بیشترین دقت مربوط به نقاط انتهایی طیف بود: کلاس ۵ با <b>۸۶.۹۷٪</b> و کلاس ۱ با <b>۷۷.۵۲٪</b>؛ جایی که بار کلمات احساسی قطعی‌تر است.</li>
                <li>کلاس‌های میانی بیشترین چالش را داشتند: کلاس ۲ (<b>۵۶.۴۲٪</b>)، کلاس ۳ (<b>۶۳.۰۳٪</b>) و کلاس ۴ (<b>۶۷.۵۸٪</b>).</li>
                <li>تحلیل خطاها نشان داد که انحراف مدل کاملاً ماهیت همسایگی دارد؛ عمده خطاها بین کلاس‌های مجاور (مانند اشتباه کلاس ۴ با ۵ یا ۲ با ۱) رخ داده و خطای شدید میان دو قطب احساسی به صفر نزدیک بوده است.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_single('new_nlp1.jpg', 'RoBERTa-large Model Results Overview')

        st.markdown("---")

        st.markdown("### 🔍 Future Direction: Aspect Detection with DistilBERT")
        st.markdown("""
        <div class="warning-card" style="direction: rtl; text-align: right;">
            <h4 style="margin-top: 0;">💡 یک ایده‌ی جدید برای آینده</h4>
            <p style="font-size: 1rem; line-height: 1.9;">
            فرض کنید می‌خواهید محصولی بخرید و بین هزاران نظر، دنبال کسی می‌گردید که چند ماه از آن استفاده کرده
            یا آن را با محصول رقیب مقایسه کرده باشد. امتیاز ستاره‌ای به‌تنهایی این اطلاعات را به ما نمی‌دهد.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            ما می‌خواستیم این جست‌وجو را ساده‌تر کنیم؛ مدلی بسازیم که تشخیص دهد نویسنده نظر سؤال می‌پرسد،
            خرید یا نخریدن را توصیه می‌کند، محصول را با رقبا مقایسه می‌کند یا تجربه استفاده بلندمدت دارد.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            چون داده‌های ما این برچسب‌ها را نداشتند، ابتدا از <b>Zero-shot</b> برای تولید برچسب‌های اولیه استفاده کردیم
            و بعد با آن‌ها مدل <b>DistilBERT</b> را آموزش دادیم.
            نتیجه ارزیابی نسخه اولیه حدود <b>۴۸ درصد</b> بود و نشان داد هنوز جای کار زیادی داریم.
            </p>
            <p style="font-size: 1rem; line-height: 1.9;">
            با توجه به محدودیت زمان بوت‌کمپ، توان پردازشی و کمبود برچسب‌های انسانیِ تأییدشده،
            تصمیم گرفتیم این پروژه را حتماً بعد از بوت‌کمپ ادامه بدهیم؛
            با تمرکز بر اصلاح برچسب‌ها و ارزیابی دقیق‌تر، تا خروجی برای تصمیم‌گیری خریداران قابل‌اعتمادتر شود.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # TAB 4: HMM SEQUENCE MODELING
    # ==========================================
    with nlp_tab4:
        st.markdown("<h2 style='text-align: right; color: #C084FC;'>🔗 HMM Sequence Modeling (Bonus)</h2>", unsafe_allow_html=True)

        # ========================================
        # Part A: Data Preprocessing
        # ========================================
        st.markdown("### ⚖️ Step 1: Class Imbalance & Random Undersampling")

        st.markdown("""
        <div class="danger-card" style="direction: rtl; text-align: right;">
            <h3 style="margin-top: 0;">⚠️ Problem: Imbalanced Class Distribution</h3>
            <p style="font-size: 1.05rem; line-height: 1.8;">
            در بررسی اولیه داده‌ها، توزیع نظرات مثبت و منفی نامتوازن بود؛ به‌طوری‌که تعداد نظرات مثبت به‌مراتب بیشتر از نظرات منفی بود.
            این عدم توازن می‌توانست باعث ایجاد سوگیری در فرایند آموزش مدل شود.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="safe-card" style="direction: rtl; text-align: right;">
            <h3 style="margin-top: 0;">✅ Solution: Random Undersampling</h3>
            <p style="font-size: 1.05rem; line-height: 1.8;">
            به‌منظور کاهش اثر عدم توازن کلاس‌ها، از روش نمونه‌گیری تصادفی (Random Undersampling) استفاده شد.
            در این مرحله، تعداد نمونه‌های هر دو کلاس به ۱۵ هزار نظر کاهش یافت و یک مجموعه داده متوازن با مجموع ۳۰ هزار نظر ایجاد شد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_pair(
            'n4.jpg', 'n5.jpg',
            'Before: Imbalanced Distribution',
            'After: Balanced (15,000 per class → 30,000 total)'
        )

        st.markdown("---")

        # --- Step 2: Review Length ---
        st.markdown("### 📏 Step 2: Review Length Analysis")

        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.8;">
            در این بخش، طول نظرات کاربران بر اساس تعداد کلمات مورد بررسی قرار گرفته است.
            توزیع طول نظرات نشان می‌دهد که بیشتر کاربران نظرات نسبتاً کوتاهی ثبت کرده‌اند؛
            درحالی‌که تعداد محدودی از نظرات دارای طول بسیار زیادی هستند.
            </p>
            <p style="font-size: 1rem; line-height: 1.8;">
            نمودار جعبه‌ای نیز پراکندگی طول نظرات و مقادیر پرت را در دو گروه مثبت و منفی مقایسه می‌کند.
            این بررسی به شناخت ویژگی‌های متنی داده‌ها و درک بهتر ساختار ورودی مدل کمک می‌کند.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_pair(
            'n6.jpg', 'n7.jpg',
            'Word Count Distribution (Histogram)',
            'Boxplot: Positive vs Negative Review Length'
        )

        st.markdown("---")

        # --- Step 3: Top Frequent Words ---
        st.markdown("### 🔤 Step 3: Top Frequent Words per Class")

        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1rem; line-height: 1.8;">
            با استخراج پرتکرارترین کلمات در هر یک از کلاس‌های احساسی، الگوی واژگانی نظرات مثبت و منفی بررسی شده است.
            این تحلیل امکان مقایسه واژگان رایج کاربران در بیان تجربه‌های مثبت و منفی را فراهم می‌کند.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_single('n8.png', 'Top Frequent Words: Positive vs Negative Sentiment')

        st.markdown("---")

        # ========================================
        # Part B: HMM Model
        # ========================================
        st.markdown("### 🔄 Step 4: HMM Transition Matrix")

        st.markdown("""
        <div class="bonus-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 1.02rem; line-height: 1.85;">
            ماتریس انتقال، احتمال جابه‌جایی بین حالت‌های متوالی (States) را در مدل HMM نشان می‌دهد.
            در این پروژه، متن هر نظر به سه حالت متوالی تقسیم شده است که به‌ترتیب بخش ابتدایی، میانی و پایانی متن را نشان می‌دهند.
            <br><br>
            نتایج نشان می‌دهد که احتمال باقی‌ماندن در هر حالت برابر با <b>۰٫۹۸۷</b> و احتمال انتقال از حالت ابتدایی به حالت بعدی برابر با <b>۰٫۰۱۳</b> است.
            همچنین، در حالت پایانی احتمال ماندن در همان حالت برابر با <b>۱</b> است.
            <br><br>
            <b>نکته‌ی مهم:</b> الگوی انتقال در هر دو کلاس مثبت و منفی یکسان مشاهده می‌شود.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_pair(
            'nlp_3.png', 'nlp_4.png',
            'HMM Transition Matrix — Positive Class',
            'HMM Transition Matrix — Negative Class'
        )

        st.markdown("---")

        # --- Markov Property ---
        st.markdown("### 🧩 Step 5: Markov Property (خاصیت مارکوف)")

        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border-right: 4px solid #C084FC;">
            <p style="font-size: 1.02rem; line-height: 1.85;">
            در مدل‌های مارکوف، پیش‌بینی وضعیت آینده بر اساس وضعیت فعلی انجام می‌شود و با دانستن حالت فعلی،
            نیازی به درنظرگرفتن تمام حالت‌های گذشته نیست. این ویژگی که به‌عنوان خاصیت مارکوف شناخته می‌شود،
            امکان تجزیه احتمال مشترک یک توالی را به حاصل‌ضرب احتمالات انتقال بین حالت‌های متوالی فراهم می‌کند.
            <br><br>
            <b>نقش در مدل HMM:</b> این فرض، ساختار وابستگی بین حالت‌های پنهان را مشخص کرده و محاسبات احتمالاتی مدل را ساده‌تر می‌کند.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_single('nlp_8.png', 'Markov Property — State Dependency Visualization')

        st.markdown("---")

        # --- Chain Rule ---
        st.markdown("### ⛓️ Step 6: Chain Rule of Probability (قانون زنجیره‌ای)")

        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right; border-right: 4px solid #C084FC;">
            <p style="font-size: 1.02rem; line-height: 1.85;">
            قانون زنجیره‌ای احتمال (Chain Rule) روشی برای محاسبه احتمال وقوع هم‌زمان چند متغیر تصادفی است.
            بر اساس این قانون، احتمال مشترک یک توالی به حاصل‌ضرب احتمال متغیر اول و احتمالات شرطی متغیرهای بعدی
            با توجه به تمام متغیرهای پیشین تجزیه می‌شود.
            <br><br>
            <b>نقش در مدل HMM:</b> این رابطه مبنای ریاضی برای بیان احتمال توالی‌هاست و با اعمال فرض مارکوف،
            امکان ساده‌سازی محاسبات احتمالاتی و مدل‌سازی توالی حالت‌های پنهان را فراهم می‌کند.
            </p>
        </div>
        """, unsafe_allow_html=True)

        show_nlp_single('nlp_9.png', 'Chain Rule — Joint Probability Factorization')

        st.markdown("---")

        # --- HMM Evaluation ---
        st.markdown("### 📊 Step 7: HMM Evaluation Results (6,000 Test Samples)")

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "84.53%")
        c2.metric("Micro F1", "84.53%")

        st.markdown("""
        <div class="bonus-card" style="direction: rtl; text-align: right;">
            <h4 style="color: #F3E8FF; margin-top: 0;">🎯 HMM Classification Performance</h4>
            <p style="font-size: 1rem; line-height: 1.85;">
            مدل HMM روی ۶۰۰۰ نمونه تست به دقت کلی <b>۸۴.۵۳ درصد</b> رسید.
            Precision و Recall در هر دو کلاس متعادل است (بازه ۰.۸۲ تا ۰.۸۷)،
            که نشان می‌دهد مدل در تشخیص هر دو کلاس عملکرد یکنواختی دارد.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Classification Report")
        hmm_report = pd.DataFrame({
            'Class': ['Negative (1-3)', 'Positive (4-5)'],
            'Precision': [0.87, 0.83],
            'Recall': [0.82, 0.87],
            'F1-Score': [0.84, 0.85],
            'Support': [3000, 3000]
        })
        st.dataframe(hmm_report, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="glass-card" style="direction: rtl; text-align: right;">
            <p style="font-size: 0.98rem; line-height: 1.8;">
            <b>📌 Interpretation:</b>
            کلاس Negative (۱-۳) و Positive (۴-۵) هر دو دقت مشابهی دارند.
            مدل HMM به عنوان یک رویکرد ساده‌ی احتمالاتی، عملکرد قابل قبولی در تفکیک احساس نظرات دارد.
            این نتیجه نشان می‌دهد که ویژگی‌های ترتیبی متن (توالی حالت‌ها) اطلاعات مفیدی برای تشخیص احساس حمل می‌کنند.
            </p>
        </div>
        """, unsafe_allow_html=True)
