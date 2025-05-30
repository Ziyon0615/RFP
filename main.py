from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image
import base64
import sqlite3
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import schedule
import threading
import time
import traceback
import uvicorn
from typing import Optional

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect('yield_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS historical_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  width REAL,
                  height REAL,
                  healthy_area REAL,
                  medium_area REAL,
                  unhealthy_area REAL,
                  predicted_yield REAL,
                  actual_yield REAL,
                  location TEXT,
                  model_type TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Model training function
def train_model():
    try:
        print("Starting yield prediction model training...")
        conn = sqlite3.connect('yield_data.db')
        df = pd.read_sql_query("SELECT * FROM historical_data WHERE actual_yield IS NOT NULL", conn)
        conn.close()
        
        if len(df) > 10:  # Only retrain with sufficient data
            X = df[['healthy_area', 'medium_area', 'unhealthy_area', 'width', 'height']]
            y = df['actual_yield']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, 'yield_model.pkl')
            print(f"Yield prediction model trained with {len(df)} records")
            return True
        else:
            print("Insufficient data for yield prediction training (min 10 records needed)")
            return False
    except Exception as e:
        print(f"Yield prediction model training failed: {str(e)}")
        traceback.print_exc()
        return False

# Schedule daily retraining
def scheduler_thread():
    schedule.every().day.at("02:00").do(train_model)
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start scheduler in background thread
threading.Thread(target=scheduler_thread, daemon=True).start()

# Try to load existing models
try:
    model = joblib.load('yield_model.pkl')
    print("Loaded trained yield prediction model")
except:
    model = None
    print("No trained yield prediction model available, using linear model")

def is_valid_image_type(content_type: str, filename: str) -> bool:
    """Check if file is JPG, JPEG or PNG"""
    valid_types = ["image/jpeg", "image/jpg", "image/png"]
    valid_extensions = [".jpg", ".jpeg", ".png"]
    
    if content_type.lower() in valid_types:
        return True
    
    if any(filename.lower().endswith(ext) for ext in valid_extensions):
        return True
    
    return False

def is_rice_field(image_np: np.ndarray) -> bool:
    """Verify if an image contains a rice field using color analysis"""
    try:
        hsv = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.count_nonzero(green_mask) / (image_np.shape[0] * image_np.shape[1]) * 100
        
        return green_percentage > 30
    except Exception as e:
        print(f"Error in rice field detection: {str(e)}")
        traceback.print_exc()
        return False

def process_image(image_data: bytes, field_width_m: float, field_height_m: float):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(image).astype(np.float32)
        
        if not is_rice_field(img_np):
            return {"error": "Uploaded image does not appear to be a rice field. Please upload a valid rice field image."}

        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        pseudo_ndvi = (G - R) / (G + R + 1e-6)

        mask_healthy = pseudo_ndvi > 0.2
        mask_medium = (pseudo_ndvi <= 0.2) & (pseudo_ndvi > 0)
        mask_unhealthy = pseudo_ndvi <= 0

        out_img = np.zeros_like(img_np)
        out_img[mask_healthy] = [0, 255, 0]
        out_img[mask_medium] = [255, 255, 0]
        out_img[mask_unhealthy] = [255, 0, 0]

        total_pixels = img_np.shape[0] * img_np.shape[1]
        m2_per_pixel = (field_width_m * field_height_m) / total_pixels
        
        healthy_area = np.sum(mask_healthy) * m2_per_pixel
        medium_area = np.sum(mask_medium) * m2_per_pixel
        unhealthy_area = np.sum(mask_unhealthy) * m2_per_pixel

        # Calculate yield using model if available, else linear model
        model_type = "linear"
        if model:
            input_data = [[healthy_area, medium_area, unhealthy_area, field_width_m, field_height_m]]
            yield_kg = model.predict(input_data)[0]
            model_type = "ML"
        else:
            yield_kg = (healthy_area * 0.8 + medium_area * 0.4 + unhealthy_area * 0.1)

        output_pil = Image.fromarray(out_img.astype(np.uint8))
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG")
        processed_image_b64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "processed_image": processed_image_b64,
            "estimated_yield": yield_kg,
            "model_type": model_type,
            "stats": {
                "healthy": round(healthy_area, 2),
                "medium": round(medium_area, 2),
                "unhealthy": round(unhealthy_area, 2)
            }
        }
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        traceback.print_exc()
        return {"error": f"Image processing failed: {str(e)}"}

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    width: float = Form(...),
    height: float = Form(...),
    location: Optional[str] = Form(None)
):
    try:
        if not is_valid_image_type(image.content_type, image.filename):
            return JSONResponse(
                status_code=400,
                content={"error": "Only JPG, JPEG, and PNG files are allowed"}
            )
            
        contents = await image.read()
        result = process_image(contents, width, height)
        
        if "error" in result:
            return JSONResponse(
                status_code=400,
                content={"error": result["error"]}
            )
        
        # Save to historical database
        conn = sqlite3.connect('yield_data.db')
        c = conn.cursor()
        c.execute('''INSERT INTO historical_data 
                    (timestamp, width, height, healthy_area, medium_area, 
                     unhealthy_area, predicted_yield, location, model_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (datetime.now(), width, height, 
                  result["stats"]["healthy"], result["stats"]["medium"], result["stats"]["unhealthy"],
                  result["estimated_yield"], location, result["model_type"]))
        conn.commit()
        lastrowid = c.lastrowid
        conn.close()
        
        # Add record ID to response
        result["record_id"] = lastrowid
            
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing image: {str(e)}"}
        )

@app.post("/save_actual_yield")
async def save_actual_yield(data: dict):
    try:
        conn = sqlite3.connect('yield_data.db')
        c = conn.cursor()
        c.execute('''UPDATE historical_data 
                    SET actual_yield = ?
                    WHERE id = ?''',
                 (data["actualYield"], data["record_id"]))
        conn.commit()
        conn.close()
        
        # Trigger immediate training if we have enough data
        if train_model():
            # Reload model if training was successful
            try:
                global model
                model = joblib.load('yield_model.pkl')
                print("Reloaded trained model after update")
            except:
                model = None
                print("Failed to reload model after update")
        
        return {"status": "success"}
    except Exception as e:
        print(f"Error saving actual yield: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Error saving data: {str(e)}"}
        )

# Endpoint for fetching historical data
@app.get("/history")
async def get_history(limit: int = Query(5, gt=0, le=50)):
    try:
        conn = sqlite3.connect('yield_data.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''SELECT id, timestamp, location, predicted_yield, actual_yield, model_type 
                     FROM historical_data 
                     ORDER BY timestamp DESC
                     LIMIT ?''', (limit,))
        rows = c.fetchall()
        conn.close()
        
        # Convert rows to list of dictionaries
        history = [dict(row) for row in rows]
        
        return history
    except Exception as e:
        print(f"Error fetching history: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Error fetching history: {str(e)}"}
        )

if __name__ == "__main__":
    # Initial training if data available
    try:
        train_model()
    except Exception as e:
        print(f"Initial training failed: {str(e)}")
        traceback.print_exc()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)