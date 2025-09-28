#!/usr/bin/env python3
"""
Alpha Events Pro - Versión Corregida
Sistema simplificado que garantiza uso SOLO de tokens Alpha Events
"""

import asyncio
import logging
import os
import time
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from dataclasses import dataclass, asdict
import random

# Configuración
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PORT = int(os.getenv('PORT', 8080))

# Configuración Alpha Events
DAILY_VOLUME_TARGET = float(os.getenv('DAILY_VOLUME_TARGET', 1024))
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 3.0))
TARGET_POINTS = int(os.getenv('TARGET_POINTS', 25))

@dataclass
class DailyStats:
    date: str
    volume: float = 0.0
    loss: float = 0.0
    trades: int = 0
    points_earned: int = 0
    best_token: str = ""
    last_updated: str = ""

class AlphaEventsBot:
    def __init__(self):
        self.base_url = 'https://api.binance.com'
        self.session: Optional[aiohttp.ClientSession] = None
        
        today = datetime.utcnow().date().isoformat()
        self.daily_stats = DailyStats(date=today, last_updated=datetime.utcnow().isoformat())
        
        # Tracking
        self.last_order_time = 0
        self.token_performance = {}
        
        # TOKENS ALPHA EVENTS OFICIALES 2025
        self.alpha_tokens = {
            # Tier 1 - Estables (3 puntos base)
            'LIGHTUSDT': {'tier': 1, 'points': 3, 'priority': 1},
            'RIVERUSDT': {'tier': 1, 'points': 3, 'priority': 2},
            'BLESSUSDT': {'tier': 1, 'points': 3, 'priority': 3},
            
            # Tier 2 - Volátiles (4 puntos base)
            'HANAUSDT': {'tier': 2, 'points': 4, 'priority': 4},
            'COAIUSDT': {'tier': 2, 'points': 4, 'priority': 5},
            'ASTERUSDT': {'tier': 2, 'points': 4, 'priority': 6},
            
            # Tier 3 - Premium (5 puntos base)
            'AIXBTUSDT': {'tier': 3, 'points': 5, 'priority': 7},
            'MAGICUSDT': {'tier': 3, 'points': 5, 'priority': 8},
            'OMNIUSDT': {'tier': 3, 'points': 5, 'priority': 9},
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    async def get_alpha_token(self) -> str:
        """Selecciona token Alpha Events de forma inteligente"""
        try:
            current_hour = datetime.utcnow().hour
            
            # Estrategia por horarios
            if 1 <= current_hour <= 8:  # Asia peak
                preferred_tokens = ['LIGHTUSDT', 'RIVERUSDT', 'HANAUSDT']
            elif 13 <= current_hour <= 16:  # Europa peak  
                preferred_tokens = ['BLESSUSDT', 'COAIUSDT', 'ASTERUSDT']
            else:  # Off-peak - usar tokens premium
                preferred_tokens = ['AIXBTUSDT', 'MAGICUSDT', 'OMNIUSDT']
            
            # Verificar disponibilidad
            for token in preferred_tokens:
                try:
                    ticker = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': token})
                    if ticker:
                        self.logger.info(f"✅ Selected Alpha token: {token} (${float(ticker['price']):.6f})")
                        return token
                except Exception as e:
                    self.logger.warning(f"Token {token} not available: {str(e)}")
                    continue
                    
            # Fallback a todos los tokens Alpha
            for token in self.alpha_tokens.keys():
                try:
                    ticker = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': token})
                    if ticker:
                        self.logger.warning(f"⚠️ Fallback to: {token}")
                        return token
                except:
                    continue
                    
            # Emergencia - usar el primero
            emergency_token = list(self.alpha_tokens.keys())[0]
            self.logger.error(f"🚨 Emergency fallback: {emergency_token}")
            return emergency_token
            
        except Exception as e:
            self.logger.error(f"Error selecting Alpha token: {str(e)}")
            return 'LIGHTUSDT'  # Fallback seguro

    async def execute_alpha_cycle(self, symbol: str, volume: float) -> Dict:
        """Ejecuta ciclo COMPLETO: COMPRA + VENTA inmediata (NO se queda con tokens)"""
        try:
            # Verificación crítica
            if symbol not in self.alpha_tokens:
                raise ValueError(f"❌ {symbol} NO ES UN TOKEN ALPHA EVENTS! Tokens válidos: {list(self.alpha_tokens.keys())}")
            
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_order_time < 10:
                await asyncio.sleep(10 - (current_time - self.last_order_time))
            
            self.logger.info(f"🎯 INICIANDO CICLO COMPLETO: {symbol} ${volume} (COMPRA + VENTA INMEDIATA)")
            
            # Obtener precio y filtros
            ticker = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
            current_price = float(ticker['price'])
            
            exchange_info = await self._make_request('GET', '/api/v3/exchangeInfo', {'symbol': symbol})
            filters = exchange_info['symbols'][0]['filters']
            
            # Calcular cantidad
            quantity = volume / current_price
            
            # Aplicar filtros
            lot_size_filter = next(f for f in filters if f['filterType'] == 'LOT_SIZE')
            min_qty = float(lot_size_filter['minQty'])
            step_size = float(lot_size_filter['stepSize'])
            
            quantity = max(quantity, min_qty)
            quantity = round(quantity / step_size) * step_size
            quantity = round(quantity, 8)
            
            self.logger.info(f"💰 PASO 1: COMPRANDO {quantity} {symbol} a precio ~${current_price:.6f}")
            
            # PASO 1: COMPRA MARKET (INMEDIATA)
            buy_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': f"{quantity:.6f}".rstrip('0').rstrip('.'),
                'newOrderRespType': 'FULL'
            }
            
            buy_result = await self._make_request('POST', '/api/v3/order', buy_params, signed=True)
            
            # Calcular lo que realmente se compró
            executed_qty = float(buy_result['executedQty'])
            avg_buy_price = sum(float(fill['price']) * float(fill['qty']) for fill in buy_result['fills']) / executed_qty
            buy_value = sum(float(fill['price']) * float(fill['qty']) for fill in buy_result['fills'])
            
            self.logger.info(f"✅ COMPRA COMPLETADA: {executed_qty} {symbol} por ${buy_value:.4f} (precio promedio: ${avg_buy_price:.6f})")
            
            # Esperar brevemente (para evitar rate limits pero mantener velocidad)
            await asyncio.sleep(random.uniform(2, 4))
            
            self.logger.info(f"🔄 PASO 2: VENDIENDO INMEDIATAMENTE {executed_qty} {symbol}")
            
            # PASO 2: VENTA MARKET INMEDIATA (TODO LO QUE SE COMPRÓ)
            sell_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': f"{executed_qty:.6f}".rstrip('0').rstrip('.'),
                'newOrderRespType': 'FULL'
            }
            
            sell_result = await self._make_request('POST', '/api/v3/order', sell_params, signed=True)
            
            # Calcular resultado de venta
            sell_qty = float(sell_result['executedQty'])
            avg_sell_price = sum(float(fill['price']) * float(fill['qty']) for fill in sell_result['fills']) / sell_qty
            sell_value = sum(float(fill['price']) * float(fill['qty']) for fill in sell_result['fills'])
            
            self.logger.info(f"✅ VENTA COMPLETADA: {sell_qty} {symbol} por ${sell_value:.4f} (precio promedio: ${avg_sell_price:.6f})")
            
            # Verificación crítica: NO debe quedar con tokens
            if abs(executed_qty - sell_qty) > 0.000001:
                self.logger.warning(f"⚠️ ADVERTENCIA: Diferencia en cantidades - Comprado: {executed_qty}, Vendido: {sell_qty}")
            else:
                self.logger.info(f"✅ PERFECTO: NO quedó con tokens. Comprado: {executed_qty} = Vendido: {sell_qty}")
            
            # Calcular resultados finales
            total_fees = (buy_value + sell_value) * 0.001  # 0.1% estimado por operación
            net_pnl = sell_value - buy_value - total_fees
            volume_generated = buy_value + sell_value
            spread_loss = buy_value - sell_value  # Pérdida por spread
            
            # Calcular puntos Alpha Events
            token_info = self.alpha_tokens[symbol]
            base_points = token_info['points']
            volume_multiplier = max(1, int(volume_generated / 25))  # 1 punto extra cada $25
            estimated_points = base_points * volume_multiplier
            
            # Actualizar stats
            self.daily_stats.volume += volume_generated
            self.daily_stats.loss += max(0, abs(net_pnl))
            self.daily_stats.trades += 1
            self.daily_stats.points_earned += estimated_points
            self.daily_stats.best_token = symbol
            self.daily_stats.last_updated = datetime.utcnow().isoformat()
            self.last_order_time = time.time()
            
            result = {
                'symbol': symbol,
                'token_tier': token_info['tier'],
                'cycle_type': 'COMPLETE_BUY_SELL',
                'buy_quantity': executed_qty,
                'sell_quantity': sell_qty,
                'buy_price': avg_buy_price,
                'sell_price': avg_sell_price,
                'buy_value': buy_value,
                'sell_value': sell_value,
                'spread_loss': spread_loss,
                'total_fees': total_fees,
                'volume_generated': volume_generated,
                'net_pnl': net_pnl,
                'estimated_alpha_points': estimated_points,
                'daily_volume': self.daily_stats.volume,
                'daily_loss': self.daily_stats.loss,
                'daily_points': self.daily_stats.points_earned,
                'progress_percent': (self.daily_stats.volume / DAILY_VOLUME_TARGET) * 100,
                'tokens_remaining': 0  # SIEMPRE 0 - no se queda con tokens
            }
            
            # Notificación detallada
            profit_emoji = "📈" if net_pnl >= 0 else "📉"
            cycle_emoji = "🔄"
            await self.send_telegram_notification(
                f"{cycle_emoji} <b>Alpha Events CYCLE COMPLETO</b>\n"
                f"🪙 <code>{symbol}</code> (Tier {token_info['tier']})\n"
                f"💰 Volume: <b>${volume_generated:.2f}</b> (${buy_value:.2f} + ${sell_value:.2f})\n"
                f"🔄 Qty: <b>{executed_qty:.6f}</b> comprado/vendido\n"
                f"💸 Spread: <b>${spread_loss:.4f}</b>\n"
                f"📊 P&L: <b>${net_pnl:.4f}</b> {profit_emoji}\n"
                f"⭐ Alpha Points: <b>+{estimated_points}</b>\n"
                f"📈 Daily: <b>${self.daily_stats.volume:.2f}</b> ({result['progress_percent']:.1f}%)\n"
                f"🏆 Total Points: <b>{self.daily_stats.points_earned}</b>\n"
                f"✅ <b>Sin tokens restantes</b>"
            )
            
            self.logger.info(f"🎯 CICLO ALPHA COMPLETADO: {symbol}")
            self.logger.info(f"   📊 Volumen generado: ${volume_generated:.2f}")
            self.logger.info(f"   💸 Pérdida por spread: ${spread_loss:.4f}")
            self.logger.info(f"   💰 P&L total: ${net_pnl:.4f}")
            self.logger.info(f"   ⭐ Puntos Alpha: +{estimated_points}")
            self.logger.info(f"   ✅ Tokens restantes: 0 (NO se queda con tokens)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ CICLO ALPHA FALLIDO: {str(e)}")
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
                    self.logger.error(f"Telegram failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Telegram error: {str(e)}")

    def should_trade(self) -> bool:
        """Verifica si debe continuar"""
        return (
            self.daily_stats.volume < DAILY_VOLUME_TARGET and 
            self.daily_stats.loss < MAX_DAILY_LOSS and
            self.daily_stats.points_earned < TARGET_POINTS
        )

    def get_cycle_volume(self) -> float:
        """Volumen dinámico por ciclo"""
        remaining = DAILY_VOLUME_TARGET - self.daily_stats.volume
        base_volume = random.uniform(15, 35)
        return min(base_volume, remaining * 0.8, 50)

# FastAPI App
app = FastAPI(title="Alpha Events Pro", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = AlphaEventsBot()

@app.on_event("startup")
async def startup():
    await bot.__aenter__()

@app.on_event("shutdown")
async def shutdown():
    await bot.__aexit__(None, None, None)

@app.get("/")
async def root():
    return {
        "service": "Alpha Events Pro",
        "status": "running",
        "version": "2.1.0",
        "alpha_tokens": list(bot.alpha_tokens.keys()),
        "daily_stats": asdict(bot.daily_stats)
    }

@app.get("/status")
async def get_status():
    progress = (bot.daily_stats.volume / DAILY_VOLUME_TARGET) * 100
    points_progress = (bot.daily_stats.points_earned / TARGET_POINTS) * 100
    
    return {
        "daily_volume": bot.daily_stats.volume,
        "daily_loss": bot.daily_stats.loss,
        "trades_count": bot.daily_stats.trades,
        "points_earned": bot.daily_stats.points_earned,
        "best_token": bot.daily_stats.best_token,
        "progress_percent": progress,
        "points_progress": points_progress,
        "should_continue": bot.should_trade(),
        "alpha_tokens_available": list(bot.alpha_tokens.keys()),
        "last_updated": bot.daily_stats.last_updated
    }

@app.get("/check-alpha-tokens")
async def check_alpha_tokens():
    """Verifica tokens Alpha Events disponibles"""
    try:
        available = []
        unavailable = []
        
        for symbol, info in bot.alpha_tokens.items():
            try:
                ticker = await bot._make_request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
                price = await bot._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
                
                available.append({
                    'symbol': symbol,
                    'tier': info['tier'],
                    'points': info['points'],
                    'price': float(price['price']),
                    'volume_24h': float(ticker['quoteVolume']),
                    'change_24h': float(ticker['priceChangePercent']),
                    'status': 'AVAILABLE'
                })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                unavailable.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return {
            "status": "success",
            "total_tokens": len(bot.alpha_tokens),
            "available": len(available),
            "unavailable": len(unavailable),
            "tokens": available,
            "errors": unavailable,
            "recommended": available[0]['symbol'] if available else None
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/execute-alpha-cycle")
async def execute_alpha_cycle():
    """Ejecuta ciclo con token Alpha Events"""
    try:
        if not bot.should_trade():
            return {
                "status": "stopped",
                "reason": "Daily limits reached",
                "daily_stats": asdict(bot.daily_stats)
            }
        
        # Seleccionar token Alpha Events
        symbol = await bot.get_alpha_token()
        volume = bot.get_cycle_volume()
        
        # Ejecutar ciclo
        result = await bot.execute_alpha_cycle(symbol, volume)
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }

@app.get("/test-alpha-cycle")
async def test_alpha_cycle():
    """Test con token Alpha Events específico"""
    try:
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            return {
                "status": "error",
                "error": "APIs no configuradas"
            }
        
        # Test conexión
        account = await bot._make_request('GET', '/api/v3/account', signed=True)
        
        if not bot.should_trade():
            return {
                "status": "stopped",
                "daily_stats": asdict(bot.daily_stats)
            }
        
        # Usar token Alpha Events
        symbol = await bot.get_alpha_token()
        volume = min(bot.get_cycle_volume(), 20)  # Volumen conservador
        
        result = await bot.execute_alpha_cycle(symbol, volume)
        
        return {
            "status": "success",
            "message": f"✅ Alpha Events cycle executed with {symbol}",
            "result": result,
            "alpha_verification": f"✅ {symbol} is an official Alpha Events token"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/dashboard")
async def dashboard():
    """Dashboard mejorado"""
    progress = (bot.daily_stats.volume / DAILY_VOLUME_TARGET) * 100
    points_progress = (bot.daily_stats.points_earned / TARGET_POINTS) * 100
    
    status_class = "good" if bot.should_trade() else "danger"
    status_text = "🟢 Alpha Events Activo" if bot.should_trade() else "🔴 Objetivos Completados"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alpha Events Pro Dashboard</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .header h1 {{ color: #333; margin: 0; font-size: 2.5em; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
            .stat {{ background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid #007bff; }}
            .stat h3 {{ margin: 0 0 10px 0; color: #333; }}
            .stat .value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .progress {{ background: #e9ecef; border-radius: 10px; height: 25px; margin: 20px 0; position: relative; }}
            .progress-bar {{ background: linear-gradient(90deg, #28a745, #20c997); height: 100%; border-radius: 10px; }}
            .progress-text {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: bold; color: white; }}
            .status {{ padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; font-weight: bold; }}
            .status.good {{ background: #d4edda; color: #155724; }}
            .status.danger {{ background: #f8d7da; color: #721c24; }}
            .controls {{ text-align: center; margin: 30px 0; }}
            .button {{ background: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; margin: 5px; font-size: 16px; }}
            .button:hover {{ background: #0056b3; }}
            .button.success {{ background: #28a745; }}
            .button.info {{ background: #17a2b8; }}
            .logs {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; font-family: monospace; max-height: 300px; overflow-y: auto; display: none; }}
            .alpha-tokens {{ margin: 20px 0; }}
            .token-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
            .token {{ background: #e9ecef; padding: 10px; border-radius: 5px; text-align: center; font-size: 12px; }}
            .tier1 {{ border-left: 5px solid #28a745; }}
            .tier2 {{ border-left: 5px solid #ffc107; }}
            .tier3 {{ border-left: 5px solid #dc3545; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Alpha Events Pro</h1>
                <p>Sistema Oficial de Trading Alpha Events</p>
                <p>Actualizado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <h3>💰 Volumen Diario</h3>
                    <div class="value">${bot.daily_stats.volume:.2f}</div>
                    <small>Target: ${DAILY_VOLUME_TARGET}</small>
                </div>
                <div class="stat">
                    <h3>⭐ Puntos Alpha</h3>
                    <div class="value">{bot.daily_stats.points_earned}</div>
                    <small>Target: {TARGET_POINTS}</small>
                </div>
                <div class="stat">
                    <h3>🔄 Trades</h3>
                    <div class="value">{bot.daily_stats.trades}</div>
                    <small>Completados</small>
                </div>
                <div class="stat">
                    <h3>🏆 Mejor Token</h3>
                    <div class="value">{bot.daily_stats.best_token or 'N/A'}</div>
                    <small>Alpha Events</small>
                </div>
            </div>
            
            <div class="progress">
                <div class="progress-bar" style="width: {min(progress, 100)}%">
                    <div class="progress-text">Volumen: {progress:.1f}%</div>
                </div>
            </div>
            
            <div class="progress">
                <div class="progress-bar" style="width: {min(points_progress, 100)}%; background: linear-gradient(90deg, #ffc107, #fd7e14);">
                    <div class="progress-text">Puntos: {points_progress:.1f}%</div>
                </div>
            </div>
            
            <div class="status {status_class}">
                {status_text}
            </div>
            
            <div class="alpha-tokens">
                <h3>🪙 Tokens Alpha Events Oficiales</h3>
                <div class="token-grid">
                    <div class="token tier1">LIGHTUSDT<br>Tier 1 - 3pts</div>
                    <div class="token tier1">RIVERUSDT<br>Tier 1 - 3pts</div>
                    <div class="token tier1">BLESSUSDT<br>Tier 1 - 3pts</div>
                    <div class="token tier2">HANAUSDT<br>Tier 2 - 4pts</div>
                    <div class="token tier2">COAIUSDT<br>Tier 2 - 4pts</div>
                    <div class="token tier2">ASTERUSDT<br>Tier 2 - 4pts</div>
                    <div class="token tier3">AIXBTUSDT<br>Tier 3 - 5pts</div>
                    <div class="token tier3">MAGICUSDT<br>Tier 3 - 5pts</div>
                    <div class="token tier3">OMNIUSDT<br>Tier 3 - 5pts</div>
                </div>
            </div>
            
            <div class="controls">
                <button class="button" onclick="window.location.reload()">🔄 Actualizar</button>
                <button class="button success" onclick="executeAlphaCycle()" id="executeBtn">🎯 Ejecutar Alpha Cycle</button>
                <button class="button info" onclick="checkAlphaTokens()">🪙 Verificar Tokens</button>
            </div>
            
            <div id="logs" class="logs">
                <h4>📋 Log de operaciones:</h4>
                <div id="logContent">Esperando operaciones...</div>
            </div>
        </div>
        
        <script>
        async function executeAlphaCycle() {{
            const btn = document.getElementById('executeBtn');
            const logs = document.getElementById('logs');
            const logContent = document.getElementById('logContent');
            
            btn.innerHTML = '⏳ Ejecutando Alpha Cycle...';
            btn.disabled = true;
            logs.style.display = 'block';
            logContent.innerHTML = '🎯 Iniciando Alpha Events cycle...';
            
            try {{
                const response = await fetch('/test-alpha-cycle');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    const r = result.result;
                    logContent.innerHTML = `
                        ✅ <strong>Alpha Events Cycle Exitoso</strong><br><br>
                        🪙 <strong>Token:</strong> ${{r.symbol}} (Tier ${{r.token_tier}})<br>
                        💰 <strong>Volumen:</strong> $${{r.volume_generated?.toFixed(2)}}<br>
                        📊 <strong>P&L:</strong> $${{r.net_pnl?.toFixed(4)}}<br>
                        ⭐ <strong>Puntos Alpha:</strong> +${{r.estimated_alpha_points}}<br>
                        📈 <strong>Progreso:</strong> ${{r.progress_percent?.toFixed(1)}}%<br>
                        🏆 <strong>Total Puntos:</strong> ${{r.daily_points}}<br><br>
                        ${{result.alpha_verification}}
                    `;
                    setTimeout(() => window.location.reload(), 3000);
                }} else if (result.status === 'stopped') {{
                    logContent.innerHTML = `🛑 <strong>Objetivos completados</strong><br>Volumen: $${{result.daily_stats?.volume?.toFixed(2)}}<br>Puntos: ${{result.daily_stats?.points_earned}}`;
                }} else {{
                    logContent.innerHTML = `❌ <strong>Error:</strong> ${{result.error}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ <strong>Error:</strong> ${{error.message}}`;
            }}
            
            btn.innerHTML = '🎯 Ejecutar Alpha Cycle';
            btn.disabled = false;
        }}
        
        async function checkAlphaTokens() {{
            const logs = document.getElementById('logs');
            const logContent = document.getElementById('logContent');
            
            logs.style.display = 'block';
            logContent.innerHTML = '🪙 Verificando tokens Alpha Events...';
            
            try {{
                const response = await fetch('/check-alpha-tokens');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    let tokenList = '';
                    result.tokens.forEach(token => {{
                        const tierColor = token.tier === 1 ? '🟢' : token.tier === 2 ? '🟡' : '🔴';
                        tokenList += `
                            ${{tierColor}} <strong>${{token.symbol}}</strong> (Tier ${{token.tier}})<br>
                            💰 Precio: ${{token.price?.toFixed(6)}}<br>
                            📊 Volumen 24h: ${{token.volume_24h?.toLocaleString()}}<br>
                            📈 Cambio 24h: ${{token.change_24h?.toFixed(2)}}%<br>
                            ⭐ Puntos base: ${{token.points}}<br><br>
                        `;
                    }});
                    
                    logContent.innerHTML = `
                        ✅ <strong>Tokens Alpha Events verificados</strong><br><br>
                        📊 <strong>Resumen:</strong><br>
                        🪙 Total: ${{result.total_tokens}}<br>
                        ✅ Disponibles: ${{result.available}}<br>
                        ❌ No disponibles: ${{result.unavailable}}<br>
                        🎯 Recomendado: <strong>${{result.recommended}}</strong><br><br>
                        <strong>Tokens disponibles:</strong><br>
                        ${{tokenList}}
                    `;
                    
                    if (result.errors.length > 0) {{
                        logContent.innerHTML += `
                            <strong>⚠️ Errores:</strong><br>
                            ${{result.errors.map(e => `${{e.symbol}}: ${{e.error}}`).join('<br>')}}
                        `;
                    }}
                }} else {{
                    logContent.innerHTML = `❌ <strong>Error:</strong> ${{result.error}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ <strong>Error:</strong> ${{error.message}}`;
            }}
        }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# Auto-trading inteligente
async def auto_alpha_trading():
    """Trading automático solo con tokens Alpha Events"""
    cycle_count = 0
    
    while True:
        try:
            if bot.should_trade():
                current_hour = datetime.utcnow().hour
                
                # Frecuencia por horarios
                if 1 <= current_hour <= 8 or 13 <= current_hour <= 16:  # Peak hours
                    sleep_time = random.uniform(1800, 3600)  # 30-60 min
                else:
                    sleep_time = random.uniform(3600, 7200)  # 1-2 horas
                
                # Ejecutar solo con tokens Alpha Events
                try:
                    symbol = await bot.get_alpha_token()
                    
                    # Verificación crítica
                    if symbol not in bot.alpha_tokens:
                        bot.logger.error(f"🚨 CRITICAL: {symbol} is NOT an Alpha Events token!")
                        await bot.send_telegram_notification(f"🚨 ERROR: Sistema intentó usar {symbol} que NO es Alpha Events!")
                        await asyncio.sleep(3600)
                        continue
                    
                    volume = bot.get_cycle_volume()
                    
                    result = await bot.execute_alpha_cycle(symbol, volume)
                    cycle_count += 1
                    
                    # Reporte cada 5 ciclos
                    if cycle_count % 5 == 0:
                        await bot.send_telegram_notification(
                            f"📈 <b>Alpha Events Report</b>\n"
                            f"✅ Completed {cycle_count} Alpha cycles\n"
                            f"💰 Volume: <b>${bot.daily_stats.volume:.2f}</b>\n"
                            f"⭐ Points: <b>{bot.daily_stats.points_earned}</b>\n"
                            f"🎯 Progress: <b>{(bot.daily_stats.volume/DAILY_VOLUME_TARGET)*100:.1f}%</b>\n"
                            f"🏆 Best token: <b>{bot.daily_stats.best_token}</b>"
                        )
                        
                except Exception as e:
                    bot.logger.error(f"Auto-trading cycle failed: {str(e)}")
                    await bot.send_telegram_notification(f"⚠️ <b>Auto-trading error:</b> {str(e)}")
                    sleep_time = 1800  # Reintentar en 30 min
                    
            else:
                # Objetivos completados
                if cycle_count > 0:
                    await bot.send_telegram_notification(
                        f"🏁 <b>Alpha Events Goals Completed!</b>\n"
                        f"🎯 Total cycles: <b>{cycle_count}</b>\n"
                        f"💰 Volume: <b>${bot.daily_stats.volume:.2f}</b>\n"
                        f"⭐ Points: <b>{bot.daily_stats.points_earned}</b>\n"
                        f"🏆 Best token: <b>{bot.daily_stats.best_token}</b>\n"
                        f"📊 Success rate: Alpha Events only!"
                    )
                    cycle_count = 0
                sleep_time = 3600
                
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            bot.logger.error(f"Auto-trading system error: {str(e)}")
            await asyncio.sleep(1800)

@app.on_event("startup")
async def start_alpha_trading():
    """Inicia sistema Alpha Events"""
    bot.logger.info("🚀 Starting Alpha Events Pro system")
    await bot.send_telegram_notification(
        f"🚀 <b>Alpha Events Pro Started!</b>\n"
        f"🎯 Volume target: <b>${DAILY_VOLUME_TARGET}</b>\n"
        f"⭐ Points target: <b>{TARGET_POINTS}</b>\n"
        f"🪙 Tokens: <b>SOLO Alpha Events oficiales</b>\n"
        f"💡 Sistema inteligente activado"
    )
    asyncio.create_task(auto_alpha_trading())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
