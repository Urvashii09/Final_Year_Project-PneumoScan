import os
import io
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
from reportlab.pdfgen import canvas

# --- App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here')

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Load Pretrained CNN Model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'pneu.cnn.model.h5')
model = load_model(model_path)

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# --- History Model ---
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(300), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- Create Database ---
with app.app_context():
    db.create_all()

# --- User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists!')
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Signup successful. Please log in.')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = os.path.join('static', imagefile.filename)
        imagefile.save(image_path)

        img = load_img(image_path, target_size=(500, 500), color_mode='grayscale')
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        probability = float(model.predict(x)[0][0])
        is_positive = probability >= 0.5
        label = "Positive for Pneumonia" if is_positive else "Healthy (Negative)"
        confidence = probability * 100 if is_positive else (1 - probability) * 100
        prediction_text = f"{label} ({confidence:.2f}%)"

        session['diagnosis'] = prediction_text

        new_record = History(user_id=current_user.id, image_path=image_path, result=prediction_text)
        db.session.add(new_record)
        db.session.commit()

        return render_template('index.html', prediction=prediction_text, imagePath=image_path,
                               confidence=confidence, is_positive=is_positive)

    return render_template('index.html')

@app.route('/generate-pdf')
@login_required
def generate_pdf():
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 750, "Patient Prediction Report")
    p.drawString(100, 720, f"User: {current_user.username}")
    diagnosis = session.get('diagnosis', 'N/A')
    p.drawString(100, 690, f"Diagnosis: {diagnosis}")
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')

@app.route('/history')
@login_required
def history():
    records = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template('history.html', records=records)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/contact_submit', methods=['POST'])
def contact_submit():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    if not name or not email or not message:
        flash("Please fill all fields")
        return redirect(url_for('contact'))

    print(f"Message from {name} ({email}): {message}")
    flash("Message sent successfully!")
    return redirect(url_for('contact'))

@app.route('/doctors')
@login_required
def doctors():
    return render_template('doctor.html')

# --- Main ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
