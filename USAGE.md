# Face Verification System - Usage Guide

## Quick Start

### 1. Start the API Server

```bash
# Make sure you're in the project directory
cd /Users/aditya/proj/face

# Run with uv
uv run python main.py

# Or activate virtual environment and run
python main.py
```

The server will start at `http://localhost:8000`

### 2. Open the Test App

Open `test_app.html` in your browser (Chrome/Firefox recommended for camera access).

**Note:** The test app is in `.gitignore` and won't be committed to git.

### 3. Test the System

#### Register a User
1. Go to the "Register" tab
2. Enter a User ID (e.g., `user123`)
3. Enter a Full Name (e.g., `John Doe`)
4. Click "Start Camera"
5. Position your face in the camera
6. Click "Capture & Register"

#### Verify a User
1. Go to the "Verify" tab
2. Enter the User ID you want to verify
3. Click "Start Camera"
4. Position your face in the camera
5. Click "Capture & Verify"
6. See the results with similarity score and liveness detection

#### Identify a User (1:N)
1. Go to the "Identify" tab
2. Click "Start Camera"
3. Position your face in the camera
4. Click "Capture & Identify"
5. The system will identify which registered user you are

#### View Users
1. Go to the "Users" tab
2. Click "Refresh List"
3. See all registered users with their embedding counts

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Register User
```bash
curl -X POST "http://localhost:8000/register" \
  -F "user_id=user123" \
  -F "name=John Doe" \
  -F "image=@face.jpg"
```

### Verify User
```bash
curl -X POST "http://localhost:8000/verify" \
  -F "user_id=user123" \
  -F "image=@face.jpg"
```

### Identify User
```bash
curl -X POST "http://localhost:8000/identify" \
  -F "image=@face.jpg"
```

### List Users
```bash
curl http://localhost:8000/users
```

### Get User Details
```bash
curl http://localhost:8000/users/user123
```

### Add Face to Profile (Enriched Embeddings)
```bash
curl -X POST "http://localhost:8000/add-face/user123" \
  -F "image=@face2.jpg"
```

### Compare Two Faces
```bash
curl -X POST "http://localhost:8000/compare" \
  -F "image1=@face1.jpg" \
  -F "image2=@face2.jpg"
```

## Features Demonstrated

✅ **Face Detection** - MTCNN automatically detects faces  
✅ **Liveness Detection** - Prevents photo spoofing  
✅ **Enriched Embeddings** - Multiple face images improve accuracy  
✅ **Real-time Camera** - Live camera feed in browser  
✅ **Beautiful UI** - Modern, responsive design  
✅ **DuckDB Storage** - Efficient database for embeddings  

## Troubleshooting

### Camera Not Working
- Make sure you're using HTTPS or localhost
- Grant camera permissions in browser
- Try Chrome or Firefox (best compatibility)

### API Offline
- Make sure the server is running (`python main.py`)
- Check that port 8000 is not in use
- Verify dependencies are installed (`uv sync`)

### Liveness Detection Failing
- Ensure good lighting
- Use a real camera (not a photo)
- Face the camera directly
- Avoid screens or printed photos

## Configuration

Edit `config.py` to adjust:
- Verification thresholds
- Liveness detection sensitivity
- Embedding enrichment settings
- API host/port

## Next Steps

- [ ] Test with multiple users
- [ ] Try different lighting conditions
- [ ] Test liveness detection with photos
- [ ] Add multiple faces per user for enriched embeddings
- [ ] Convert model to TFLite for mobile deployment
