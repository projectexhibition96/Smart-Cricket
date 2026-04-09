# Smart Cricket

AI-powered cricket shot analytics with a futuristic dashboard, CSV/JSON pose ingestion, a Flask model service, and a 3D batting visualizer.

## Monorepo structure

- `client/`: React (Vite) + Tailwind + Framer Motion + React Three Fiber
- `server/`: Node.js + Express + JWT + MongoDB (Atlas) + upload proxy to Flask
- `python-service/`: Flask service that loads a scikit-learn `.pkl` and returns predictions
- `uploads/`: temporary uploads (created automatically)
- `client/public/models/`: your `.glb/.gltf` batting models + `manifest.json`

## Prerequisites

- Node.js 20+ (recommended)
- Python 3.10+
- MongoDB Atlas connection string

## 1) Python service setup

From `python-service/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The Flask service runs at `http://localhost:5001`.

## 2) Server setup (Express)

Create `server/.env`:

```bash
PORT=5050
MONGODB_URI="YOUR_ATLAS_CONNECTION_STRING"
JWT_SECRET="change-me"
PYTHON_SERVICE_URL="http://localhost:5001"
CLIENT_ORIGIN="http://localhost:5173,http://127.0.0.1:5173"
UPLOAD_DIR="../uploads"
```

Run:

```bash
cd server
npm install
npm run dev
```

Server runs at `http://localhost:5050` (port **5050** avoids macOS AirPlay using **5000**).

## 3) Client setup (Vite)

Create `client/.env`:

```bash
VITE_API_BASE_URL="http://localhost:5050"
```

Run:

```bash
cd client
npm install
npm run dev
```

Client runs at `http://localhost:5173`.

## Using the AI Shot Classifier (live webcam)

1. Put **`shot_model.pkl`** (your joblib model from `live_feedback.py`) in `python-service/`.
2. Optionally copy **`ideal_batting_angles.json.example`** → **`ideal_batting_angles.json`** and tune angles (same format as your desktop script).
3. Reinstall Python deps (`pip install -r requirements.txt`) so **MediaPipe** + **OpenCV** are available.
4. Run Flask, then the Node API, then the Vite client. Open **AI Shot Classifier**, allow the camera, then **Start classification**.

Frames are sent from the browser to the API → Flask **`/predict_frame`** → MediaPipe pose + your model (same 33×4 landmark features as `live_feedback.py`).

The **CSV / batch `/predict` endpoint** is still available for offline datasets if you call it directly.

## 3D model labels

Your model may output labels like `cover_drive`. Add those exact strings to `predictionLabels` in `client/public/models/manifest.json` so the 3D lab picks the right GLB.

## 3D models

Place your models in `client/public/models/`.

Update `client/public/models/manifest.json` to include your files and which prediction label they correspond to.

## Notes on feature validation

The Flask service validates:

- File type is CSV/JSON
- Columns match the model’s expected feature names (uses `model.feature_names_in_` when available)
- Clear errors returned with missing/extra columns and the expected list

