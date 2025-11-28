"""
Client application for Credit Card Fraud Detection
Serves the front-end HTML and acts as a proxy to the model server
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import requests
import os
from datetime import datetime

app = FastAPI(title="Credit Card Fraud Detection Client", version="1.0.0")

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_SERVER_URL = "http://localhost:5000"  # URL of the model server
CLIENT_PORT = 8080  # Port for the client server

# Pydantic models for request/response validation
class TransactionRequest(BaseModel):
    amount_float: float = Field(..., description="Transaction amount")
    amount_is_refund: int = Field(0, ge=0, le=1, description="Is refund flag")
    avg_transaction_per_card_norm: float = Field(..., description="Normalized avg transaction per card")
    avg_transaction_per_user_norm: float = Field(..., description="Normalized avg transaction per user")
    country_development: str = Field(..., description="Country development status")
    transaction_datetime: str = Field(..., description="Transaction date and time")
    mcc: int = Field(..., description="Merchant Category Code")
    errbin_insufficient_balance: int = Field(0, ge=0, le=1)
    errbin_technical_error: int = Field(0, ge=0, le=1)
    errbin_authentication_error: int = Field(0, ge=0, le=1)
    errbin_card_error: int = Field(0, ge=0, le=1)
    has_error: int = Field(0, ge=0, le=1)
    online_flag: int = Field(0, ge=0, le=1)
    usechip_swipe_transaction: int = Field(0, ge=0, le=1)
    usechip_chip_transaction: int = Field(0, ge=0, le=1)

class FeedbackRequest(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID")
    is_correct: bool = Field(..., description="Whether the prediction was correct")

# Read the HTML file
def load_html():
    html_path = os.path.join(os.path.dirname(__file__), 'front_end.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.get('/', response_class=HTMLResponse)
async def index():
    """Serve the main HTML page"""
    html_content = load_html()
    return HTMLResponse(content=html_content)

@app.post('/predict')
async def predict(transaction_data: TransactionRequest):
    """
    Forward prediction request to the model server
    Receives transaction features from front-end and sends to server
    """
    try:
        
        # Prepare data for model server
        # Convert country_development to numeric encoding
        country_mapping = {
            'developed': 2,
            'developing': 1,
            'underdeveloped': 0
        }
        
        # Extract date/time components if needed
        transaction_datetime = transaction_data.transaction_datetime
        dt = None
        if transaction_datetime:
            try:
                # Handle ISO format with or without timezone
                dt_str = transaction_datetime.replace('Z', '+00:00')
                dt = datetime.fromisoformat(dt_str)
            except:
                # Try parsing without timezone
                dt = datetime.fromisoformat(transaction_datetime)
        
        # Prepare feature vector for model
        model_features = {
            'Amount_float': transaction_data.amount_float,
            'Amount_IsRefund': transaction_data.amount_is_refund,
            'avg_transaction_per_card_norm': transaction_data.avg_transaction_per_card_norm,
            'avg_transaction_per_user_norm': transaction_data.avg_transaction_per_user_norm,
            'country_development': country_mapping.get(transaction_data.country_development, 1),
            'MCC': transaction_data.mcc,
            'ErrBin_InsufficientBalance': transaction_data.errbin_insufficient_balance,
            'ErrBin_TechnicalError': transaction_data.errbin_technical_error,
            'ErrBin_AuthenticationError': transaction_data.errbin_authentication_error,
            'ErrBin_CardError': transaction_data.errbin_card_error,
            'HasError': transaction_data.has_error,
            'OnlineFlag': transaction_data.online_flag,
            'UseChip_SwipeTransaction': transaction_data.usechip_swipe_transaction,
            'UseChip_ChipTransaction': transaction_data.usechip_chip_transaction,
        }
        
        # Add datetime features if available
        if dt:
            model_features['transaction_hour'] = dt.hour
            model_features['transaction_day'] = dt.day
            model_features['transaction_month'] = dt.month
            model_features['transaction_year'] = dt.year
        
        # Forward request to model server
        try:
            response = requests.post(
                f"{MODEL_SERVER_URL}/predict",
                json=model_features,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'prediction': result.get('prediction'),
                    'confidence': result.get('confidence', 0.0),
                    'transaction_id': result.get('transaction_id')
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f'Model server error: {response.text}'
                )
                
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail='Cannot connect to model server. Please ensure server.py is running on port 5000.'
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail='Request to model server timed out.'
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Error communicating with model server: {str(e)}'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Error processing request: {str(e)}'
        )

@app.post('/feedback')
async def feedback(feedback_data: FeedbackRequest):
    """
    Forward feedback to the model server
    Receives user feedback about prediction accuracy
    """
    try:
        
        # Forward feedback to model server
        try:
            response = requests.post(
                f"{MODEL_SERVER_URL}/feedback",
                json={
                    'transaction_id': feedback_data.transaction_id,
                    'is_correct': feedback_data.is_correct
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'message': result.get('message', 'Feedback received')
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f'Model server error: {response.text}'
                )
                
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail='Cannot connect to model server. Please ensure server.py is running on port 5000.'
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Error communicating with model server: {str(e)}'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f'Error processing feedback: {str(e)}'
        )

@app.get('/health')
async def health():
    """Health check endpoint"""
    try:
        # Check if model server is reachable
        response = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        model_server_status = "connected" if response.status_code == 200 else "disconnected"
    except:
        model_server_status = "disconnected"
    
    return {
        'status': 'healthy',
        'client_port': CLIENT_PORT,
        'model_server': model_server_status,
        'model_server_url': MODEL_SERVER_URL
    }

if __name__ == '__main__':
    import uvicorn
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║   Credit Card Fraud Detection - Client Server            ║
    ╠══════════════════════════════════════════════════════════╣
    ║   Client running on: http://localhost:{CLIENT_PORT}               ║
    ║   Model server URL: {MODEL_SERVER_URL}                ║
    ║   API docs: http://localhost:{CLIENT_PORT}/docs                   ║
    ║                                                          ║
    ║   Open your browser and navigate to:                     ║
    ║   http://localhost:{CLIENT_PORT}                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host='0.0.0.0', port=CLIENT_PORT, reload=False)

