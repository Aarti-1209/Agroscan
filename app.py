from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os, requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'agroscan-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agroscan.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['WEATHER_API_KEY'] = os.environ.get('WEATHER_API_KEY')
app.config['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY')
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    language = db.Column(db.String(5), default='en')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scans = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200))
    disease_name = db.Column(db.String(200))
    class_name = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    severity = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

model = tf.keras.models.load_model('model/plant_disease.h5')
with open('model/class_names.json') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_lang():
    lang = session.get('lang', 'en')
    if current_user.is_authenticated and hasattr(current_user, 'language'):
        lang = current_user.language
    try:
        with open(f'languages/{lang}.json', encoding='utf-8') as f:
            return json.load(f)
    except:
        with open('languages/en.json', encoding='utf-8') as f:
            return json.load(f)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def calculate_risks_and_tips(temp, humidity, rain, clouds, wind):
    risks = []
    tips = []

    if humidity > 80 and temp > 20:
        risks.append({'disease': 'Fungal Disease (Fungas Rog)', 'level': 'High', 'reason': f'Humidity {humidity}% + Temperature {temp}°C', 'crops': 'Tomato, Potato, Wheat, Rice'})
        tips.append({'title': 'Prevent Fungal Disease', 'icon': '🍄', 'level': 'High', 'actions': ['Spray Mancozeb or Copper fungicide immediately', 'Avoid overhead irrigation — use drip instead', 'Ensure good air circulation between plants', 'Remove infected leaves and destroy them', 'Apply fungicide every 7 days in humid weather']})

    if rain > 0 or (humidity > 85 and temp < 25):
        risks.append({'disease': 'Late Blight (Pichli Jhulsa)', 'level': 'High', 'reason': f'Rainfall detected + High humidity', 'crops': 'Tomato, Potato'})
        tips.append({'title': 'Prevent Late Blight', 'icon': '🌧️', 'level': 'High', 'actions': ['Apply Metalaxyl + Mancozeb spray before rain', 'After rain — spray Chlorothalonil immediately', 'Remove water from field quickly', 'Do not work in field when plants are wet', 'Use resistant varieties next season']})

    if temp > 25 and humidity > 60 and humidity < 80:
        risks.append({'disease': 'Early Blight (Agati Jhulsa)', 'level': 'Medium', 'reason': f'Warm temperature {temp}°C with moderate humidity {humidity}%', 'crops': 'Tomato, Potato, Brinjal'})
        tips.append({'title': 'Prevent Early Blight', 'icon': '🍅', 'level': 'Medium', 'actions': ['Spray Copper fungicide every 10 days', 'Keep plants well-spaced for air flow', 'Remove lower infected leaves', 'Mulch around plants to prevent soil splash', 'Water plants at base, not on leaves']})

    if temp > 30 and humidity < 50:
        risks.append({'disease': 'Aphids & Sucking Pests', 'level': 'Medium', 'reason': f'Hot ({temp}°C) and dry ({humidity}% humidity)', 'crops': 'Cotton, Mustard, Vegetables'})
        tips.append({'title': 'Control Aphids & Pests', 'icon': '🐛', 'level': 'Medium', 'actions': ['Spray Neem oil solution (5ml per liter water)', 'Use yellow sticky traps in field', 'Spray Imidacloprid if infestation is heavy', 'Encourage natural predators (ladybugs)', 'Check plants early morning for pest signs']})

    if clouds > 60 and humidity > 60 and temp > 20 and rain == 0:
        risks.append({'disease': 'Powdery Mildew', 'level': 'Medium', 'reason': f'Cloudy {clouds}% + Humid {humidity}% without rain', 'crops': 'Grapes, Peas, Cucumber, Wheat'})
        tips.append({'title': 'Prevent Powdery Mildew', 'icon': '⬜', 'level': 'Medium', 'actions': ['Spray Sulphur-based fungicide', 'Apply Potassium Bicarbonate spray', 'Improve ventilation around plants', 'Avoid excessive nitrogen fertilizer', 'Use resistant crop varieties']})

    if wind > 40:
        risks.append({'disease': 'Strong Wind — Crop Damage Risk', 'level': 'High', 'reason': f'Wind speed {wind} km/h', 'crops': 'All crops especially tall plants'})
        tips.append({'title': 'Protect from Strong Wind', 'icon': '💨', 'level': 'High', 'actions': ['Provide support/stakes to tall plants', 'Harvest ready crops immediately', 'Create windbreaks using trees or nets', 'Avoid spraying pesticides in strong wind', 'Check irrigation systems for damage after wind']})

    if not risks:
        risks.append({'disease': 'All Clear — Good Weather! 🌟', 'level': 'Low', 'reason': 'Current weather conditions are favorable for crops', 'crops': 'All crops'})
        tips.append({'title': 'Maintain Good Practices', 'icon': '✅', 'level': 'Low', 'actions': ['Continue regular monitoring of crops', 'Apply balanced NPK fertilizer', 'Maintain proper irrigation schedule', 'Keep field clean of weeds', 'This is good time for spraying preventive fungicides']})

    return risks, tips

# ─── Routes ───────────────────────────────────────

@app.route('/')
def home():
    return redirect(url_for('language'))

@app.route('/language')
def language():
    return render_template('language.html')

@app.route('/set-language', methods=['POST'])
def set_language():
    lang = request.form.get('lang', 'en')
    session['lang'] = lang
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    t = get_lang()
    if request.method == 'POST':
        phone = request.form.get('phone')
        password = request.form.get('password')
        user = User.query.filter_by(phone=phone).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['lang'] = user.language
            return redirect(url_for('dashboard'))
        flash('Invalid phone number or password!')
    return render_template('login.html', t=t)

@app.route('/register', methods=['GET', 'POST'])
def register():
    t = get_lang()
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        password = request.form.get('password')
        otp = request.form.get('otp')
        lang = session.get('lang', 'en')
        if otp != '123456':
            flash('Invalid OTP! Please enter correct OTP.')
            return redirect(url_for('register'))
        if User.query.filter_by(phone=phone).first():
            flash('Phone number already registered!')
            return redirect(url_for('register'))
        user = User(name=name, phone=phone, password=generate_password_hash(password), language=lang)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('register.html', t=t)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    t = get_lang()
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.created_at.desc()).all()
    total = len(scans)
    most_common = max(set([s.disease_name for s in scans]), key=[s.disease_name for s in scans].count) if scans else None
    current_month = datetime.utcnow().month
    monthly = len([s for s in scans if s.created_at.month == current_month])
    return render_template('dashboard.html', scans=scans, total=total, user=current_user, most_common=most_common, monthly=monthly, t=t)

@app.route('/scan')
@login_required
def scan():
    t = get_lang()
    return render_template('index.html', t=t)

@app.route('/shops')
@login_required
def shops():
    t = get_lang()
    return render_template('shops.html', t=t)

@app.route('/weather')
@login_required
def weather():
    t = get_lang()
    return render_template('weather.html', t=t)

@app.route('/chatbot')
@login_required
def chatbot():
    t = get_lang()
    return render_template('chatbot.html', t=t)

@app.route('/geocode')
@login_required
def geocode():
    location = request.args.get('q')
    try:
        r = requests.get('https://nominatim.openstreetmap.org/search', params={'format': 'json', 'q': location, 'limit': 1}, headers={'User-Agent': 'AgroScan/1.0'}, timeout=10)
        return jsonify(r.json())
    except:
        return jsonify([])

@app.route('/search-shops')
@login_required
def search_shops():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    filter_type = request.args.get('filter', 'agro')
    tags = {'agro': 'shop=agrarian', 'pesticide': 'shop=agrarian', 'fertilizer': 'shop=agrarian', 'seed': 'shop=agrarian', 'garden_centre': 'shop=garden_centre'}
    tag = tags.get(filter_type, 'shop=agrarian')
    query = f'[out:json];(node[{tag}](around:5000,{lat},{lng});node[shop=garden_centre](around:5000,{lat},{lng});node[shop](around:3000,{lat},{lng}););out center 20;'
    try:
        r = requests.get('https://overpass-api.de/api/interpreter', params={'data': query}, headers={'User-Agent': 'AgroScan/1.0'}, timeout=15)
        return jsonify(r.json())
    except:
        return jsonify({'elements': []})

@app.route('/get-weather')
@login_required
def get_weather():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    key = app.config['WEATHER_API_KEY']
    try:
        weather_r = requests.get('https://api.openweathermap.org/data/2.5/weather', params={'lat': lat, 'lon': lng, 'appid': key, 'units': 'metric'}, timeout=10)
        weather = weather_r.json()
        forecast_r = requests.get('https://api.openweathermap.org/data/2.5/forecast', params={'lat': lat, 'lon': lng, 'appid': key, 'units': 'metric', 'cnt': 8}, timeout=10)
        forecast = forecast_r.json()
        temp = weather['main']['temp']
        humidity = weather['main']['humidity']
        rain = weather.get('rain', {}).get('1h', 0)
        clouds = weather['clouds']['all']
        wind = weather['wind']['speed'] * 3.6
        risks, tips = calculate_risks_and_tips(temp, humidity, rain, clouds, wind)
        return jsonify({'weather': weather, 'forecast': forecast, 'risks': risks, 'tips': tips})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-chatbot', methods=['POST'])
@login_required
def ask_chatbot():
    data = request.get_json()
    user_message = data.get('message', '')
    chat_history = data.get('history', [])

    key = app.config['GEMINI_API_KEY']

    system_prompt = """You are AgroScan AI — an expert agricultural assistant for Indian farmers.
You help farmers with crop disease identification and treatment, pesticide and fertilizer recommendations (Indian brands), weather-based farming advice, crop care tips, and general farming questions.
Always give practical, simple advice in simple English or Hindi/Hinglish as needed.
Keep responses concise and farmer-friendly.
When recommending pesticides, mention Indian brand names available in local markets.
Always end with an encouraging note for the farmer."""

    messages = []
    for h in chat_history:
        messages.append({'role': h['role'], 'parts': [{'text': h['text']}]})
    messages.append({'role': 'user', 'parts': [{'text': user_message}]})

    import time
    for attempt in range(3):
        try:
            r = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={key}',
                json={
                    'system_instruction': {'parts': [{'text': system_prompt}]},
                    'contents': messages,
                    'generationConfig': {'maxOutputTokens': 2048, 'temperature': 0.7, 'topP': 0.9}
                },
                timeout=20
            )
            result = r.json()
            if 'candidates' in result:
                reply = result['candidates'][0]['content']['parts'][0]['text']
                return jsonify({'reply': reply})
            elif 'error' in result and result['error'].get('code') == 429:
                time.sleep(2)
                continue
            else:
                return jsonify({'reply': 'Sorry, please try again!'})
        except Exception as e:
            time.sleep(1)
            continue

    return jsonify({'reply': 'Too many requests. Please wait a moment and try again!'})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Photo nahi mili'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File select karo'}), 400
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img_array = preprocess_image(filepath)
    predictions = model.predict(img_array)[0]
    top3_idx = np.argsort(predictions)[-3:][::-1]
    top3 = [{'class': idx_to_class[i], 'confidence': round(float(predictions[i]) * 100, 2)} for i in top3_idx]
    top_idx = top3_idx[0]
    class_name = idx_to_class[top_idx]
    confidence = float(predictions[top_idx])
    from disease_info import get_disease_info
    info = get_disease_info(class_name)
    new_scan = Scan(user_id=current_user.id, image_path=f'static/uploads/{filename}', disease_name=info.get('name', class_name), class_name=class_name, confidence=round(confidence * 100, 2), severity=info.get('severity', 'Unknown'))
    db.session.add(new_scan)
    db.session.commit()

    if confidence < 0.60:
        return jsonify({
            'class_name': 'uncertain',
            'confidence': round(confidence * 100, 2),
            'disease_name': 'Uncertain — Please try again with a clearer photo',
            'symptoms': 'Photo quality low ya disease unclear hai, model confident nahi hai.',
            'treatment': ['Behtar lighting mein photo lo', 'Diseased leaf ko close-up mein lo', 'Ek hi leaf ka clear photo lo, background clutter avoid karo', 'Expert agronomist se milein agar problem persist kare'],
            'prevention': 'Clear, well-lit close-up photo se better results milenge.',
            'severity': 'Unknown',
            'top3': top3,
            'image_url': f'/static/uploads/{filename}'
        })

    return jsonify({'class_name': class_name, 'confidence': round(confidence * 100, 2), 'disease_name': info.get('name', class_name), 'symptoms': info.get('symptoms', ''), 'treatment': info.get('treatment', []), 'prevention': info.get('prevention', ''), 'severity': info.get('severity', 'Unknown'), 'top3': top3, 'image_url': f'/static/uploads/{filename}'})

@app.route('/history')
@login_required
def history():
    t = get_lang()
    scans = Scan.query.filter_by(user_id=current_user.id)\
                      .order_by(Scan.created_at.desc()).all()
    return render_template('history.html', scans=scans, user=current_user, t=t)

@app.route('/scan/<int:scan_id>')
@login_required
def scan_detail(scan_id):
    t = get_lang()
    scan_record = Scan.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
    from disease_info import get_disease_info
    info = get_disease_info(scan_record.class_name)
    return render_template('scan_detail.html', scan=scan_record, info=info, t=t)

@app.route('/scan/<int:scan_id>/delete', methods=['POST'])
@login_required
def delete_scan(scan_id):
    scan_record = Scan.query.filter_by(id=scan_id, user_id=current_user.id).first_or_404()
    db.session.delete(scan_record)
    db.session.commit()
    return jsonify({'success': True})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)