"""
ws_fsl_dynamic_server.py
WebSocket server for dynamic sign language recognition
"""

import base64
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.gesture.fsl_dynamic_inference import (
    initialize_dynamic_model,
    add_frame_to_buffer,
    predict_dynamic_sign,
    reset_buffer
)

router = APIRouter()


@router.websocket("/ws/fsl-dynamic")
async def fsl_dynamic_endpoint(websocket: WebSocket):
    """WebSocket endpoint for dynamic sign recognition"""
    await websocket.accept()
    print("📱 Mobile connected to /ws/fsl-dynamic")
    
    try:
        # Initialize model
        initialize_dynamic_model()
        print("✅ Dynamic model initialized")
        
        # Reset buffer for new session
        reset_buffer()
        
        frame_count = 0
        
        while True:
            # Receive frame
            frame_b64 = await websocket.receive_text()
            
            # Decode frame
            try:
                img_bytes = base64.b64decode(frame_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        'success': False,
                        'prediction': 'UNKNOWN',
                        'confidence': 0.0,
                        'message': 'Failed to decode frame'
                    })
                    continue
                
                # Add frame to buffer
                added = add_frame_to_buffer(frame)
                
                if not added:
                    # No hands detected in this frame
                    if frame_count % 10 == 0:  # Log every 10th miss
                        print("⚠️ No hands detected in frame")
                    frame_count += 1
                    continue
                
                frame_count += 1
                
                # Predict every 5 frames (reduce computation)
                if frame_count % 5 == 0:
                    result = predict_dynamic_sign(confidence_threshold=0.7)
                    await websocket.send_json(result)
                
            except Exception as e:
                print(f"❌ Frame processing error: {e}")
                await websocket.send_json({
                    'success': False,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'message': str(e)
                })
    
    except WebSocketDisconnect:
        print("🔌 Mobile disconnected from /ws/fsl-dynamic")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close()