#!/usr/bin/env python3
"""
Alpha Events Automation para Railway
Sistema simplificado todo-en-uno
"""

import asyncio
import logging
import os
import time
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from dataclasses import dataclass, asdict

# Configuración desde variables de entorno
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PORT = int(os.getenv('PORT', 8080))

# Configuración Alpha Events
DAILY_VOLUME_TARGET = float(os.getenv('DAILY_VOLUME_TARGET', 512))
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 2.0))
TARGET_POINTS = int(os.getenv('TARGET_POINTS', 17))

@dataclass
class DailyStats:
    date: str
    volume: float = 0.0
    loss: float = 0.0
    trades: int = 0
    last_updated: str = ""

class AlphaEventsBot:
    def __init__(self):
        self.base_url = 'https://api.binance.com'
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Estado diario (en memoria - Railway es stateless)
        today = datetime.utcnow().date().isoformat()
        self.daily_stats = DailyStats(date=today, last_updated=datetime.utcnow().isoformat())
        
        # Tracking de rate limits
        self.last_order_time = 0
        self.orders_this_minute = 0
        
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AlphaEvents')

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, params: str) -> str:
        return hmac.new(
            BINANCE_SECRET_KEY.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False):
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self._generate_signature(query_string)
            params['signature'] = signature
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"Binance API error: {response.status} - {error_text}")
                    raise HTTPException(status_code=response.status, detail=error_text)
                    
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise

    async def get_optimal_token(self) -> str:
        """Selecciona el mejor token para Alpha Events según la hora"""
        current_hour = datetime.utcnow().hour
        
        # Tokens estables para menor slippage
        stable_tokens = ['ADAUSDT', 'DOTUSDT', 'MATICUSDT']
        
        # Durante horas asiáticas, tokens con más volumen
        if 1 <= current_hour <= 8:
            high_volume_tokens = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            return high_volume_tokens[current_hour % len(high_volume_tokens)]
        else:
            return stable_tokens[current_hour % len(stable_tokens)]

    async def execute_volume_cycle(self, symbol: str, target_volume: float) -> Dict:
        """Ejecuta un ciclo de compra-venta para generar volumen"""
        try:
            # Rate limiting simple
            current_time = time.time()
            if current_time - self.last_order_time < 6:  # Máximo 10 orders por minuto
                await asyncio.sleep(6 - (current_time - self.last_order_time))
            
            # Obtener precio actual
            ticker = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
            current_price = float(ticker['price'])
            
            # Calcular cantidad
            quantity = target_volume / current_price
            
            # Obtener filtros del símbolo para cantidad mínima
            exchange_info = await self._make_request('GET', '/api/v3/exchangeInfo', {'symbol': symbol})
            filters = exchange_info['symbols'][0]['filters']

            # Encontrar LOT_SIZE filter
            lot_size_filter = next(f for f in filters if f['filterType'] == 'LOT_SIZE')
            min_qty = float(lot_size_filter['minQty'])
            step_size = float(lot_size_filter['stepSize'])

            # Ajustar cantidad según filtros
            quantity = max(quantity, min_qty)
            quantity = round(quantity / step_size) * step_size
            quantity = round(quantity, 8)
            
            self.logger.info(f"Executing volume cycle: {symbol} ${target_volume}")
            
            # Orden de compra (market para velocidad)
            buy_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': f"{quantity:.6f}".rstrip('0').rstrip('.'),
                'newOrderRespType': 'FULL'
            }
            
            buy_result = await self._make_request('POST', '/api/v3/order', buy_params, signed=True)
            
            # Esperar un momento
            await asyncio.sleep(2)
            
            # Orden de venta
            executed_qty = float(buy_result['executedQty'])
            sell_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': f"{executed_qty:.6f}".rstrip('0').rstrip('.'),
                'newOrderRespType': 'FULL'
            }
            
            sell_result = await self._make_request('POST', '/api/v3/order', sell_params, signed=True)
            
            # Calcular resultados
            buy_value = sum(float(fill['price']) * float(fill['qty']) for fill in buy_result['fills'])
            sell_value = sum(float(fill['price']) * float(fill['qty']) for fill in sell_result['fills'])
            
            # Calcular fees (0.1% cada operación)
            total_fees = (buy_value + sell_value) * 0.001
            net_loss = buy_value - sell_value + total_fees
            volume_generated = buy_value + sell_value
            
            # Actualizar stats diarias
            self.daily_stats.volume += volume_generated
            self.daily_stats.loss += net_loss
            self.daily_stats.trades += 1
            self.daily_stats.last_updated = datetime.utcnow().isoformat()
            self.last_order_time = time.time()
            
            result = {
                'symbol': symbol,
                'volume_generated': volume_generated,
                'net_loss': net_loss,
                'daily_volume': self.daily_stats.volume,
                'daily_loss': self.daily_stats.loss,
                'progress_percent': (self.daily_stats.volume / DAILY_VOLUME_TARGET) * 100
            }
            
            await self.send_telegram_notification(
                f"🎯 Alpha Events Cycle\n"
                f"Symbol: {symbol}\n"
                f"Volume: ${volume_generated:.2f}\n"
                f"Loss: ${net_loss:.4f}\n"
                f"Daily: ${self.daily_stats.volume:.2f} ({result['progress_percent']:.1f}%)"
            )
            
            self.logger.info(f"Cycle completed: {symbol} - Volume: ${volume_generated:.2f}, Loss: ${net_loss:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Volume cycle failed: {str(e)}")
            raise

    async def send_telegram_notification(self, message: str):
        """Envía notificación a Telegram"""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
            
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    self.logger.error(f"Telegram notification failed: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Telegram error: {str(e)}")

    def should_trade(self) -> bool:
        """Verifica si debe continuar operando"""
        return (
            self.daily_stats.volume < DAILY_VOLUME_TARGET and 
            self.daily_stats.loss < MAX_DAILY_LOSS
        )

# FastAPI app
app = FastAPI(title="Alpha Events Automation", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del bot
bot = AlphaEventsBot()

@app.on_event("startup")
async def startup():
    await bot.__aenter__()

@app.on_event("shutdown")
async def shutdown():
    await bot.__aexit__(None, None, None)

# Endpoints API
@app.get("/")
async def root():
    return {
        "service": "Alpha Events Automation",
        "status": "running",
        "version": "1.0.0",
        "daily_stats": asdict(bot.daily_stats)
    }

@app.get("/status")
async def get_status():
    """Estado actual del sistema"""
    progress = (bot.daily_stats.volume / DAILY_VOLUME_TARGET) * 100
    return {
        "daily_volume": bot.daily_stats.volume,
        "daily_loss": bot.daily_stats.loss,
        "trades_count": bot.daily_stats.trades,
        "progress_percent": progress,
        "target_volume": DAILY_VOLUME_TARGET,
        "max_loss": MAX_DAILY_LOSS,
        "should_continue": bot.should_trade(),
        "last_updated": bot.daily_stats.last_updated
    }

@app.post("/execute-cycle")
async def execute_cycle(background_tasks: BackgroundTasks):
    """Ejecuta un ciclo de volumen"""
    if not bot.should_trade():
        return {
            "message": "Daily limits reached",
            "daily_stats": asdict(bot.daily_stats)
        }
    
    try:
        symbol = await bot.get_optimal_token()
        remaining_volume = DAILY_VOLUME_TARGET - bot.daily_stats.volume
        cycle_volume = min(remaining_volume, 25)  # Máximo $25 por ciclo
        
        result = await bot.execute_volume_cycle(symbol, cycle_volume)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/execute-cycle-test")
async def execute_cycle_test():
    """Test endpoint para ejecutar ciclo manualmente"""
    try:
        # Verificar APIs primero
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            return {
                "status": "error",
                "error": "APIs de Binance no configuradas",
                "help": "Configura BINANCE_API_KEY y BINANCE_SECRET_KEY en Railway"
            }
        
        # Test conexión a Binance
        account = await bot._make_request('GET', '/api/v3/account', signed=True)
        
        if not bot.should_trade():
            return {
                "status": "stopped",
                "reason": "Daily limits reached", 
                "daily_stats": asdict(bot.daily_stats),
                "api_connection": "OK"
            }
        
        # Intentar ejecutar ciclo
        symbol = await bot.get_optimal_token()
        remaining_volume = DAILY_VOLUME_TARGET - bot.daily_stats.volume
        cycle_volume = min(remaining_volume, 15)  # Test con volumen menor
        
        result = await bot.execute_volume_cycle(symbol, cycle_volume)
        return {
            "status": "success",
            "result": result,
            "api_connection": "OK"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "help": "Revisa que las APIs de Binance tengan permisos de trading y fondos suficientes"
        }

@app.get("/test-binance-connection")
async def test_binance_connection():
    """Test de conexión con Binance"""
    try:
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            return {
                "status": "error",
                "error": "APIs no configuradas"
            }
        
        # Test básico
        server_time = await bot._make_request('GET', '/api/v3/time')
        
        # Test con autenticación
        account = await bot._make_request('GET', '/api/v3/account', signed=True)
        
        # Obtener balances principales
        balances = {balance['asset']: float(balance['free']) 
                   for balance in account['balances'] 
                   if float(balance['free']) > 0}
        
        return {
            "status": "success",
            "server_time": server_time,
            "account_type": account.get('accountType'),
            "can_trade": account.get('canTrade'),
            "permissions": account.get('permissions'),
            "balances_with_funds": balances
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/dashboard")
async def get_dashboard():
    """Dashboard HTML con botón mejorado"""
    progress = (bot.daily_stats.volume / DAILY_VOLUME_TARGET) * 100
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alpha Events Dashboard</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat h3 {{ margin: 0 0 10px 0; color: #333; }}
            .stat .value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .progress {{ background: #e9ecef; border-radius: 10px; overflow: hidden; height: 20px; margin: 20px 0; }}
            .progress-bar {{ background: #28a745; height: 100%; transition: width 0.3s; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; text-align: center; font-weight: bold; }}
            .status.good {{ background: #d4edda; color: #155724; }}
            .status.warning {{ background: #fff3cd; color: #856404; }}
            .status.danger {{ background: #f8d7da; color: #721c24; }}
            .button {{ background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; margin: 5px; font-size: 16px; }}
            .button:hover {{ background: #0056b3; }}
            .button.success {{ background: #28a745; }}
            .button.success:hover {{ background: #1e7e34; }}
            .button.warning {{ background: #ffc107; color: #212529; }}
            .button.warning:hover {{ background: #e0a800; }}
            .logs {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; font-family: monospace; max-height: 300px; overflow-y: auto; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Alpha Events Dashboard</h1>
            <p>Actualizado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            
            <div class="stats">
                <div class="stat">
                    <h3>Volumen Diario</h3>
                    <div class="value">${bot.daily_stats.volume:.2f}</div>
                    <small>Target: ${DAILY_VOLUME_TARGET}</small>
                </div>
                <div class="stat">
                    <h3>Pérdida Diaria</h3>
                    <div class="value">${bot.daily_stats.loss:.4f}</div>
                    <small>Máximo: ${MAX_DAILY_LOSS}</small>
                </div>
                <div class="stat">
                    <h3>Operaciones</h3>
                    <div class="value">{bot.daily_stats.trades}</div>
                    <small>Completadas</small>
                </div>
                <div class="stat">
                    <h3>Progreso</h3>
                    <div class="value">{progress:.1f}%</div>
                    <small>Del target</small>
                </div>
            </div>
            
            <div class="progress">
                <div class="progress-bar" style="width: {min(progress, 100)}%"></div>
            </div>
            
            <div class="status {'good' if bot.should_trade() and bot.daily_stats.loss < 1 else 'warning' if bot.should_trade() else 'danger'}">
                {'🟢 Sistema Activo' if bot.should_trade() else '🔴 Límites Alcanzados'}
            </div>
            
            <div style="margin-top: 30px; text-align: center;">
                <button class="button" onclick="window.location.reload()">
                    🔄 Actualizar
                </button>
                <button class="button success" onclick="executeCycle()" id="executeBtn">
                    ▶️ Ejecutar Ciclo
                </button>
                <button class="button warning" onclick="testConnection()">
                    🔗 Test Conexión
                </button>
            </div>
            
            <div id="logs" class="logs" style="display: none;">
                <h4>Log de operaciones:</h4>
                <div id="logContent">Esperando operaciones...</div>
            </div>
        </div>
        
        <script>
        async function executeCycle() {{
            const btn = document.getElementById('executeBtn');
            const logs = document.getElementById('logs');
            const logContent = document.getElementById('logContent');
            
            btn.innerHTML = '⏳ Ejecutando...';
            btn.disabled = true;
            logs.style.display = 'block';
            logContent.innerHTML = 'Iniciando ciclo de volumen...';
            
            try {{
                const response = await fetch('/execute-cycle-test');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    logContent.innerHTML = `
                        ✅ Ciclo ejecutado exitosamente<br>
                        Symbol: ${{result.result.symbol}}<br>
                        Volumen: $${{result.result.volume_generated?.toFixed(2)}}<br>
                        Pérdida: $${{result.result.net_loss?.toFixed(4)}}<br>
                        Volumen diario: $${{result.result.daily_volume?.toFixed(2)}}
                    `;
                    setTimeout(() => window.location.reload(), 3000);
                }} else {{
                    logContent.innerHTML = `❌ ${{result.status}}: ${{result.error}}<br><br>💡 ${{result.help || ''}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ Error de conexión: ${{error.message}}`;
            }}
            
            btn.innerHTML = '▶️ Ejecutar Ciclo';
            btn.disabled = false;
        }}
        
        async function testConnection() {{
            const logContent = document.getElementById('logContent');
            const logs = document.getElementById('logs');
            
            logs.style.display = 'block';
            logContent.innerHTML = 'Verificando conexión con Binance...';
            
            try {{
                const response = await fetch('/test-binance-connection');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    logContent.innerHTML = `
                        ✅ Conexión exitosa<br>
                        Tipo cuenta: ${{result.account_type}}<br>
                        Puede operar: ${{result.can_trade}}<br>
                        Permisos: ${{result.permissions?.join(', ')}}<br>
                        Balances: ${{Object.entries(result.balances_with_funds || {{}}).map(([k,v]) => k + ': ' + v).join(', ')}}
                    `;
                }} else {{
                    logContent.innerHTML = `❌ Error: ${{result.error}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ Error: ${{error.message}}`;
            }}
        }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Tarea automática cada 2 horas
async def automatic_trading():
    """Tarea automática de trading"""
    while True:
        try:
            if bot.should_trade():
                symbol = await bot.get_optimal_token()
                remaining = DAILY_VOLUME_TARGET - bot.daily_stats.volume
                cycle_volume = min(remaining, 50)
                
                if cycle_volume > 5:  # Mínimo $5 por ciclo
                    await bot.execute_volume_cycle(symbol, cycle_volume)
                    
            await asyncio.sleep(7200)  # 2 horas
            
        except Exception as e:
            bot.logger.error(f"Automatic trading error: {str(e)}")
            await asyncio.sleep(600)  # Esperar 10 minutos en caso de error

@app.on_event("startup")
async def start_automatic_trading():
    """Inicia trading automático"""
    asyncio.create_task(automatic_trading())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)

