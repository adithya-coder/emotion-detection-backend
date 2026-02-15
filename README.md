# 🎭 Emotion Detection Backend

AI-powered emotion detection system with questionnaire-based risk assessment.

## 🚀 Features

- **Emotion Detection**: Detects 5 emotions (Anger, Fear, Happy, Neutral, Sad)
- **Face Detection**: Automatic face detection using Haar Cascade
- **Questionnaire Assessment**: 5-question mental health screening
- **Risk Level Classification**: Normal / Low Risk / High Risk
- **Combined Scoring**: Merges AI + questionnaire results
- **RESTful API**: Easy integration with any frontend
- **Docker Support**: Containerized deployment
- **CI/CD Ready**: GitHub Actions for auto-build & push to Docker Hub

## 📋 API Endpoints

### 1. Basic Emotion Detection
```bash
POST /detect
Content-Type: multipart/form-data

Body:
  image: <file>

Response:
{
  "Emotion": "Happy",
  "confidence": 95.67
}
```

### 2. Enhanced Detection with Questionnaire
```bash
POST /detect_with_questionnaire
Content-Type: multipart/form-data

Body:
  image: <file>
  questionnaire_score: 8.5

Response:
{
  "emotion": "Happy",
  "emotion_confidence": 95.67,
  "questionnaire_score": 8.5,
  "combined_score": 8.72,
  "risk_level": "Normal"
}
```

### 3. Health Check
```bash
GET /

Response:
{
  "status": "healthy",
  "service": "emotion-detection-backend",
  "version": "1.0.0"
}
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Using Docker CLI
```bash
# Build image
docker build -t emotion-detection-backend .

# Run container
docker run -p 5000:5000 emotion-detection-backend
```

### Pull from Docker Hub
```bash
docker pull yourusername/emotion-detection-backend:latest
docker run -p 5000:5000 yourusername/emotion-detection-backend:latest
```

## 💻 Local Development

### Prerequisites
- Python 3.10+
- Virtual environment

### Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements_fixed.txt

# Run the app
python app.py
```

Server runs at: `http://localhost:5000`

## 📁 Project Structure
```
emotion_detection_backend/
├── app.py                      # Flask application
├── requirements_fixed.txt      # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── .dockerignore              # Docker ignore rules
├── emotion_assessment.html     # Frontend demo
├── models/                     # ML model directory
│   └── face/
│       └── Face_Emotion1.h5   # Trained emotion model
└── DOCKER_SETUP.md            # CI/CD setup guide
```

## 🧪 Testing

### Using curl
```bash
curl -X POST http://localhost:5000/detect_with_questionnaire \
  -F "image=@photo.jpg" \
  -F "questionnaire_score=7.5"
```

### Using Python
```python
import requests

url = "http://localhost:5000/detect_with_questionnaire"
files = {"image": open("photo.jpg", "rb")}
data = {"questionnaire_score": 7.5}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## 📊 Risk Level Scoring

| Combined Score | Risk Level | Description |
|---------------|-----------|-------------|
| 8.0 - 10.0 | 🟢 Normal | Healthy emotional state |
| 4.0 - 7.9 | 🟡 Low Risk | Minor concerns, monitor |
| 0.0 - 3.9 | 🔴 High Risk | Requires attention |

**Calculation**: `(Questionnaire Score + Emotion Confidence/10) / 2`

## 🛠️ Technologies

- **Backend**: Flask, Flask-CORS
- **ML Framework**: TensorFlow 2.18, Keras 3.8
- **Computer Vision**: OpenCV, Pillow
- **Deployment**: Docker, Gunicorn
- **CI/CD**: GitHub Actions

## 🔒 Environment Variables

```bash
FLASK_ENV=production  # production or development
```

## 📝 Model Details

- **Architecture**: CNN (Convolutional Neural Network)
- **Input**: 48×48 RGB images
- **Classes**: 5 emotions
- **Framework**: TensorFlow/Keras
- **Model File**: `Face_Emotion1.h5`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- OpenCV for computer vision tools
- Flask team for the web framework
