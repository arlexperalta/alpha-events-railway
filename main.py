#!/usr/bin/env python3
"""
Alpha Events Pro - Sistema Optimizado para Railway
Trading automatizado con estrategias avanzadas y múltiples fuentes de ingresos
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
import math

# Configuración desde variables de entorno
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PORT = int(os.getenv('PORT', 8080))

# Configuración Alpha Events optimizada
DAILY_VOLUME_TARGET = float(os.getenv('DAILY_VOLUME_TARGET', 1024))  # Aumentado para más puntos
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 3.0))
TARGET_POINTS = int(os.getenv('TARGET_POINTS', 25))
MIN_CYCLE_VOLUME = float(os.getenv('MIN_CYCLE_VOLUME', 8))
MAX_CYCLE_VOLUME = float(os.getenv('MAX_CYCLE_VOLUME', 35))

@dataclass
class DailyStats:
    date: str
    volume: float = 0.0
    loss: float = 0.0
    trades: int = 0
    points_earned: int = 0
    best_token: str = ""
    last_updated: str = ""
    arbitrage_profit: float = 0.0
    grid_profit: float = 0.0

@dataclass
class TokenMetrics:
    symbol: str
    volume_24h: float
    price_change: float
    spread: float
    volatility: float
    liquidity_score: float
    alpha_points_potential: int

class AlphaEventsProBot:
    def __init__(self):
        self.base_url = 'https://api.binance.com'
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Estado diario mejorado
        today = datetime.utcnow().date().isoformat()
        self.daily_stats = DailyStats(date=today, last_updated=datetime.utcnow().isoformat())
        
        # Tracking avanzado
        self.last_order_time = 0
        self.orders_this_minute = 0
        self.token_performance = {}
        self.market_conditions = {}
        
        # Tokens Alpha Events con scoring actualizado 2025
        self.alpha_tokens = {
            # Tier 1 - Alto volumen y estabilidad
            'LIGHTUSDT': {'tier': 1, 'base_points': 3, 'volatility': 'medium'},
            'RIVERUSDT': {'tier': 1, 'base_points': 3, 'volatility': 'medium'},
            'BLESSUSDT': {'tier': 1, 'base_points': 3, 'volatility': 'medium'},
            
            # Tier 2 - Volumen medio, mayor volatilidad
            'HANAUSDT': {'tier': 2, 'base_points': 4, 'volatility': 'high'},
            'COAIUSDT': {'tier': 2, 'base_points': 4, 'volatility': 'high'},
            'ASTERUSDT': {'tier': 2, 'base_points': 4, 'volatility': 'high'},
            
            # Tier 3 - Nuevos tokens con mayor potencial
            'AIXBTUSDT': {'tier': 3, 'base_points': 5, 'volatility': 'very_high'},
            'MAGICUSDT': {'tier': 3, 'base_points': 5, 'volatility': 'very_high'},
            'OMNIUSDT': {'tier': 3, 'base_points': 5, 'volatility': 'very_high'},
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AlphaEventsPro')

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

    async def analyze_token_metrics(self, symbol: str) -> TokenMetrics:
        """Análisis avanzado de métricas del token"""
        try:
            # Obtener datos del ticker 24h
            ticker = await self._make_request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
            
            # Obtener orderbook para calcular spread
            orderbook = await self._make_request('GET', '/api/v3/depth', {'symbol': symbol, 'limit': 5})
            
            # Calcular métricas
            volume_24h = float(ticker['quoteVolume'])
            price_change = abs(float(ticker['priceChangePercent']))
            
            # Calcular spread
            best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else 0
            best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
            spread = ((best_ask - best_bid) / best_ask * 100) if best_ask > 0 else 0
            
            # Calcular volatilidad (high-low range)
            high = float(ticker['highPrice'])
            low = float(ticker['lowPrice'])
            volatility = ((high - low) / low * 100) if low > 0 else 0
            
            # Score de liquidez basado en volumen y orderbook
            liquidity_score = min(100, (volume_24h / 1000000) * 50 + (1 / max(spread, 0.01)) * 10)
            
            # Potencial de puntos Alpha basado en tier y condiciones
            token_info = self.alpha_tokens.get(symbol, {'tier': 3, 'base_points': 2})
            alpha_points_potential = token_info['base_points']
            
            # Bonus por condiciones favorables
            if volatility > 5:  # Alta volatilidad = más oportunidades
                alpha_points_potential += 1
            if volume_24h > 5000000:  # Alto volumen
                alpha_points_potential += 1
            if spread < 0.1:  # Spread bajo
                alpha_points_potential += 1
                
            return TokenMetrics(
                symbol=symbol,
                volume_24h=volume_24h,
                price_change=price_change,
                spread=spread,
                volatility=volatility,
                liquidity_score=liquidity_score,
                alpha_points_potential=alpha_points_potential
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return TokenMetrics(symbol, 0, 0, 0, 0, 0, 1)

    async def get_optimal_token_advanced(self) -> Tuple[str, TokenMetrics]:
        """Selección inteligente SOLO de tokens Alpha Events"""
        try:
            current_hour = datetime.utcnow().hour
            best_token = None
            best_score = 0
            best_metrics = None
            
            # SOLO tokens Alpha Events - verificación explícita
            alpha_symbols = list(self.alpha_tokens.keys())
            self.logger.info(f"Analyzing ONLY Alpha Events tokens: {alpha_symbols}")
            
            # Analizar todos los tokens Alpha Events
            for symbol in alpha_symbols:
                try:
                    # Verificar que el símbolo esté en la lista Alpha
                    if symbol not in self.alpha_tokens:
                        self.logger.warning(f"Skipping {symbol} - not in Alpha Events list")
                        continue
                        
                    metrics = await self.analyze_token_metrics(symbol)
                    
                    # Calcular score compuesto
                    score = 0
                    
                    # Factor de puntos Alpha (40% peso)
                    score += metrics.alpha_points_potential * 10
                    
                    # Factor de liquidez (25% peso)
                    score += metrics.liquidity_score * 0.25
                    
                    # Factor de volatilidad (20% peso) - más volatilidad = más oportunidades
                    score += min(metrics.volatility * 2, 20)
                    
                    # Factor de spread (15% peso) - spread bajo es mejor
                    score += max(0, (1 - metrics.spread) * 15)
                    
                    # Bonus por horarios pico
                    if 1 <= current_hour <= 8 or 13 <= current_hour <= 16:  # Asia y Europa
                        score *= 1.2
                    
                    # Bonus por performance reciente
                    if symbol in self.token_performance:
                        recent_success = self.token_performance[symbol].get('success_rate', 0)
                        score *= (1 + recent_success * 0.1)
                    
                    self.logger.info(f"Token {symbol}: score {score:.2f}, points potential {metrics.alpha_points_potential}")
                    
                    if score > best_score:
                        best_score = score
                        best_token = symbol
                        best_metrics = metrics
                        
                    await asyncio.sleep(0.2)  # Rate limiting más conservador
                    
                except Exception as e:
                    self.logger.warning(f"Could not analyze {symbol}: {str(e)}")
                    continue
            
            if best_token and best_token in self.alpha_tokens:
                self.logger.info(f"✅ Selected Alpha Events token: {best_token} with score {best_score:.2f}")
                return best_token, best_metrics
            else:
                # Fallback SOLO a tokens Alpha Events
                fallback_tokens = ['LIGHTUSDT', 'RIVERUSDT', 'BLESSUSDT']
                for fallback in fallback_tokens:
                    if fallback in self.alpha_tokens:
                        self.logger.warning(f"⚠️ Using fallback Alpha token: {fallback}")
                        try:
                            metrics = await self.analyze_token_metrics(fallback)
                            return fallback, metrics
                        except:
                            continue
                
                # Si todo falla, usar el primero de la lista
                first_alpha = list(self.alpha_tokens.keys())[0]
                self.logger.error(f"🚨 Emergency fallback to: {first_alpha}")
                return first_alpha, TokenMetrics(first_alpha, 0, 0, 0, 0, 50, 3)
                
        except Exception as e:
            self.logger.error(f"Error in Alpha token selection: {str(e)}")
            # Fallback final solo Alpha Events
            emergency_token = 'LIGHTUSDT'
            return emergency_token, TokenMetrics(emergency_token, 0, 0, 0, 0, 50, 3)

    async def execute_smart_arbitrage(self, symbol: str, metrics: TokenMetrics, volume: float) -> Dict:
        """Ejecuta arbitraje inteligente con timing optimizado"""
        try:
            # Obtener precio actual con mayor precisión
            ticker = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})
            current_price = float(ticker['price'])
            
            # Obtener orderbook para timing preciso
            orderbook = await self._make_request('GET', '/api/v3/depth', {'symbol': symbol, 'limit': 10})
            
            # Calcular cantidad óptima
            quantity = volume / current_price
            
            # Obtener filtros del símbolo
            exchange_info = await self._make_request('GET', '/api/v3/exchangeInfo', {'symbol': symbol})
            filters = exchange_info['symbols'][0]['filters']

            # Aplicar filtros
            lot_size_filter = next(f for f in filters if f['filterType'] == 'LOT_SIZE')
            min_qty = float(lot_size_filter['minQty'])
            step_size = float(lot_size_filter['stepSize'])

            quantity = max(quantity, min_qty)
            quantity = round(quantity / step_size) * step_size
            quantity = round(quantity, 8)
            
            # Timing inteligente basado en spread y volatilidad
            if metrics.spread > 0.2:  # Spread alto, esperar mejor momento
                await asyncio.sleep(random.uniform(1, 3))
            
            self.logger.info(f"Smart arbitrage: {symbol} ${volume} (Spread: {metrics.spread:.3f}%)")
            
            # Estrategia 1: Compra Market + Venta Limit (mejor para spreads grandes)
            if metrics.spread > 0.15:
                return await self._execute_market_limit_strategy(symbol, quantity, current_price, orderbook)
            # Estrategia 2: Ambas Market (mejor para alta liquidez)
            else:
                return await self._execute_dual_market_strategy(symbol, quantity)
                
        except Exception as e:
            self.logger.error(f"Smart arbitrage failed: {str(e)}")
            raise

    async def _execute_market_limit_strategy(self, symbol: str, quantity: float, price: float, orderbook: Dict) -> Dict:
        """Estrategia Market Buy + Limit Sell"""
        try:
            # Compra market
            buy_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': f"{quantity:.6f}".rstrip('0').rstrip('.'),
                'newOrderRespType': 'FULL'
            }
            
            buy_result = await self._make_request('POST', '/api/v3/order', buy_params, signed=True)
            
            # Calcular precio de venta óptimo (intentar ganar en el spread)
            avg_buy_price = sum(float(fill['price']) * float(fill['qty']) for fill in buy_result['fills']) / sum(float(fill['qty']) for fill in buy_result['fills'])
            
            # Precio de venta: entre el precio de compra y el mejor ask
            best_ask = float(orderbook['asks'][0][0])
            sell_price = min(avg_buy_price * 1.001, best_ask * 0.999)  # Pequeño profit + competitivo
            
            executed_qty = float(buy_result['executedQty'])
            
            # Venta limit
            sell_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'LIMIT',
                'quantity': f"{executed_qty:.6f}".rstrip('0').rstrip('.'),
                'price': f"{sell_price:.8f}".rstrip('0').rstrip('.'),
                'timeInForce': 'IOC',  # Immediate or Cancel
                'newOrderRespType': 'FULL'
            }
            
            sell_result = await self._make_request('POST', '/api/v3/order', sell_params, signed=True)
            
            # Si la orden limit no se ejecuta completamente, usar market para el resto
            if sell_result['status'] != 'FILLED':
                remaining_qty = float(sell_result['origQty']) - float(sell_result['executedQty'])
                if remaining_qty > 0:
                    market_sell_params = {
                        'symbol': symbol,
                        'side': 'SELL',
                        'type': 'MARKET',
                        'quantity': f"{remaining_qty:.6f}".rstrip('0').rstrip('.'),
                        'newOrderRespType': 'FULL'
                    }
                    
                    market_sell_result = await self._make_request('POST', '/api/v3/order', market_sell_params, signed=True)
                    
                    # Combinar resultados
                    sell_result['fills'].extend(market_sell_result['fills'])
                    sell_result['executedQty'] = str(float(sell_result['executedQty']) + float(market_sell_result['executedQty']))
            
            return self._calculate_trade_results(symbol, buy_result, sell_result)
            
        except Exception as e:
            self.logger.error(f"Market-Limit strategy failed: {str(e)}")
            raise

    async def _execute_dual_market_strategy(self, symbol: str, quantity: float) -> Dict:
        """Estrategia dual market (original mejorada)"""
        try:
            # Compra market
            buy_params = {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': f"{quantity:.6f}".rstrip('0').rstrip('.'),
                'newOrderRespType': 'FULL'
            }
            
            buy_result = await self._make_request('POST', '/api/v3/order', buy_params, signed=True)
            
            # Delay inteligente basado en volatilidad
            await asyncio.sleep(random.uniform(1.5, 4))
            
            # Venta market
            executed_qty = float(buy_result['executedQty'])
            sell_params = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': f"{executed_qty:.6f}".rstrip('0').rstrip('.'),
                'newOrderRespType': 'FULL'
            }
            
            sell_result = await self._make_request('POST', '/api/v3/order', sell_params, signed=True)
            
            return self._calculate_trade_results(symbol, buy_result, sell_result)
            
        except Exception as e:
            self.logger.error(f"Dual market strategy failed: {str(e)}")
            raise

    def _calculate_trade_results(self, symbol: str, buy_result: Dict, sell_result: Dict) -> Dict:
        """Calcula resultados del trade con métricas avanzadas"""
        try:
            # Calcular valores
            buy_value = sum(float(fill['price']) * float(fill['qty']) for fill in buy_result['fills'])
            sell_value = sum(float(fill['price']) * float(fill['qty']) for fill in sell_result['fills'])
            
            # Calcular fees (0.1% cada operación por defecto)
            buy_fees = sum(float(fill['commission']) for fill in buy_result['fills'] if 'commission' in fill)
            sell_fees = sum(float(fill['commission']) for fill in sell_result['fills'] if 'commission' in fill)
            total_fees = buy_fees + sell_fees
            
            # Si no hay información de fees, estimamos
            if total_fees == 0:
                total_fees = (buy_value + sell_value) * 0.001
            
            net_pnl = sell_value - buy_value - total_fees
            volume_generated = buy_value + sell_value
            
            # Calcular puntos Alpha Events estimados
            token_info = self.alpha_tokens.get(symbol, {'base_points': 2})
            estimated_points = max(1, int(volume_generated / 100) * token_info['base_points'])
            
            # Actualizar stats diarias
            self.daily_stats.volume += volume_generated
            self.daily_stats.loss += max(0, abs(net_pnl))  # Solo contar pérdidas
            self.daily_stats.trades += 1
            self.daily_stats.points_earned += estimated_points
            self.daily_stats.last_updated = datetime.utcnow().isoformat()
            
            if not self.daily_stats.best_token or estimated_points > 3:
                self.daily_stats.best_token = symbol
            
            # Actualizar performance del token
            if symbol not in self.token_performance:
                self.token_performance[symbol] = {'trades': 0, 'total_pnl': 0, 'success_rate': 0}
            
            self.token_performance[symbol]['trades'] += 1
            self.token_performance[symbol]['total_pnl'] += net_pnl
            success_rate = max(0, 1 - abs(net_pnl) / volume_generated) if volume_generated > 0 else 0
            self.token_performance[symbol]['success_rate'] = (
                self.token_performance[symbol]['success_rate'] * 0.8 + success_rate * 0.2
            )
            
            self.last_order_time = time.time()
            
            result = {
                'symbol': symbol,
                'volume_generated': volume_generated,
                'net_pnl': net_pnl,
                'total_fees': total_fees,
                'estimated_points': estimated_points,
                'daily_volume': self.daily_stats.volume,
                'daily_loss': self.daily_stats.loss,
                'daily_points': self.daily_stats.points_earned,
                'progress_percent': (self.daily_stats.volume / DAILY_VOLUME_TARGET) * 100,
                'buy_price': buy_value / sum(float(fill['qty']) for fill in buy_result['fills']),
                'sell_price': sell_value / sum(float(fill['qty']) for fill in sell_result['fills'])
            }
            
            # Notificación mejorada
            profit_emoji = "📈" if net_pnl >= 0 else "📉"
            await self.send_telegram_notification(
                f"{profit_emoji} <b>Alpha Events Pro Trade</b>\n"
                f"🪙 <code>{symbol}</code>\n"
                f"💰 Volume: <b>${volume_generated:.2f}</b>\n"
                f"📊 P&L: <b>${net_pnl:.4f}</b>\n"
                f"⭐ Points: <b>+{estimated_points}</b>\n"
                f"📈 Daily: <b>${self.daily_stats.volume:.2f}</b> ({result['progress_percent']:.1f}%)\n"
                f"🎯 Total Points: <b>{self.daily_stats.points_earned}</b>"
            )
            
            self.logger.info(f"Trade completed: {symbol} - Volume: ${volume_generated:.2f}, P&L: ${net_pnl:.4f}, Points: +{estimated_points}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating results: {str(e)}")
            raise

    async def send_telegram_notification(self, message: str):
        """Envía notificación a Telegram con HTML formatting"""
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
        """Verifica si debe continuar operando con lógica mejorada"""
        # Verificar límites básicos
        if self.daily_stats.volume >= DAILY_VOLUME_TARGET:
            return False
            
        if self.daily_stats.loss >= MAX_DAILY_LOSS:
            return False
        
        # Verificar si ya alcanzamos suficientes puntos
        if self.daily_stats.points_earned >= TARGET_POINTS:
            return False
            
        # Rate limiting inteligente
        time_since_last = time.time() - self.last_order_time
        if time_since_last < 10:  # Mínimo 10 segundos entre trades
            return False
            
        return True

    def get_dynamic_volume(self) -> float:
        """Calcula volumen dinámico basado en progreso y condiciones"""
        remaining_volume = DAILY_VOLUME_TARGET - self.daily_stats.volume
        remaining_ratio = remaining_volume / DAILY_VOLUME_TARGET
        
        # Volumen base
        if remaining_ratio > 0.8:  # Inicio del día
            base_volume = random.uniform(MIN_CYCLE_VOLUME, MIN_CYCLE_VOLUME + 10)
        elif remaining_ratio > 0.5:  # Medio día
            base_volume = random.uniform(MIN_CYCLE_VOLUME + 5, MAX_CYCLE_VOLUME - 5)
        elif remaining_ratio > 0.2:  # Final del día
            base_volume = random.uniform(MAX_CYCLE_VOLUME - 10, MAX_CYCLE_VOLUME)
        else:  # Últimos trades
            base_volume = min(remaining_volume * 0.8, MAX_CYCLE_VOLUME)
        
        # Ajustar por loss ratio
        loss_ratio = self.daily_stats.loss / MAX_DAILY_LOSS if MAX_DAILY_LOSS > 0 else 0
        if loss_ratio > 0.7:  # Reducir volumen si las pérdidas son altas
            base_volume *= 0.7
        
        return max(MIN_CYCLE_VOLUME, min(base_volume, remaining_volume, MAX_CYCLE_VOLUME))

# FastAPI app con mejoras
app = FastAPI(title="Alpha Events Pro", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia global del bot
bot = AlphaEventsProBot()

@app.on_event("startup")
async def startup():
    await bot.__aenter__()

@app.on_event("shutdown")
async def shutdown():
    await bot.__aexit__(None, None, None)

# Endpoints API mejorados
@app.get("/")
async def root():
    return {
        "service": "Alpha Events Pro",
        "status": "running",
        "version": "2.0.0",
        "daily_stats": asdict(bot.daily_stats),
        "features": [
            "Smart Token Selection",
            "Advanced Arbitrage Strategies", 
            "Dynamic Volume Optimization",
            "Real-time Market Analysis",
            "Multi-tier Token Support"
        ]
    }

@app.get("/status")
async def get_status():
    """Estado avanzado del sistema"""
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
        "target_volume": DAILY_VOLUME_TARGET,
        "target_points": TARGET_POINTS,
        "max_loss": MAX_DAILY_LOSS,
        "should_continue": bot.should_trade(),
        "next_volume": bot.get_dynamic_volume(),
        "token_performance": bot.token_performance,
        "last_updated": bot.daily_stats.last_updated
    }

@app.post("/execute-smart-cycle")
async def execute_smart_cycle(background_tasks: BackgroundTasks):
    """Ejecuta ciclo inteligente optimizado"""
    if not bot.should_trade():
        return {
            "message": "Trading stopped - limits reached",
            "daily_stats": asdict(bot.daily_stats),
            "reason": "Volume target, loss limit, or points target reached"
        }
    
    try:
        # Selección inteligente de token
        symbol, metrics = await bot.get_optimal_token_advanced()
        
        # Volumen dinámico
        cycle_volume = bot.get_dynamic_volume()
        
        # Ejecutar arbitraje inteligente
        result = await bot.execute_smart_arbitrage(symbol, metrics, cycle_volume)
        
        return {
            "status": "success",
            "result": result,
            "token_analysis": {
                "selected_token": symbol,
                "volume_24h": f"${metrics.volume_24h:,.2f}",
                "spread": f"{metrics.spread:.3f}%",
                "volatility": f"{metrics.volatility:.2f}%",
                "liquidity_score": f"{metrics.liquidity_score:.1f}/100",
                "alpha_points_potential": metrics.alpha_points_potential
            },
            "api_connection": "OK"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "help": "Revisa que las APIs de Binance tengan permisos de trading y fondos suficientes"
        }

@app.get("/analyze-tokens")
async def analyze_all_tokens():
    """Análisis completo de todos los tokens Alpha Events"""
    try:
        analysis_results = []
        
        for symbol in bot.alpha_tokens.keys():
            try:
                metrics = await bot.analyze_token_metrics(symbol)
                token_info = bot.alpha_tokens[symbol]
                
                analysis_results.append({
                    "symbol": symbol,
                    "tier": token_info['tier'],
                    "base_points": token_info['base_points'],
                    "volume_24h": metrics.volume_24h,
                    "volatility": metrics.volatility,
                    "spread": metrics.spread,
                    "liquidity_score": metrics.liquidity_score,
                    "alpha_points_potential": metrics.alpha_points_potential,
                    "performance": bot.token_performance.get(symbol, {})
                })
                
                await asyncio.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                analysis_results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        # Ordenar por potencial de puntos
        analysis_results.sort(key=lambda x: x.get('alpha_points_potential', 0), reverse=True)
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "total_tokens": len(analysis_results),
            "analysis": analysis_results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/check-alpha-tokens")
async def check_alpha_tokens():
    """Verificar disponibilidad de tokens Alpha Events en Binance"""
    try:
        available_tokens = []
        unavailable_tokens = []
        
        for symbol in bot.alpha_tokens.keys():
            try:
                # Verificar si el símbolo existe en Binance
                ticker = await bot._make_request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
                exchange_info = await bot._make_request('GET', '/api/v3/exchangeInfo', {'symbol': symbol})
                
                if ticker and exchange_info:
                    symbol_info = exchange_info['symbols'][0]
                    is_trading = symbol_info['status'] == 'TRADING'
                    
                    available_tokens.append({
                        'symbol': symbol,
                        'status': symbol_info['status'],
                        'is_trading': is_trading,
                        'volume_24h': float(ticker['quoteVolume']),
                        'price_change': float(ticker['priceChangePercent']),
                        'tier': bot.alpha_tokens[symbol]['tier'],
                        'base_points': bot.alpha_tokens[symbol]['base_points']
                    })
                else:
                    unavailable_tokens.append(symbol)
                    
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                unavailable_tokens.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return {
            "status": "success",
            "total_alpha_tokens": len(bot.alpha_tokens),
            "available_count": len(available_tokens),
            "unavailable_count": len(unavailable_tokens),
            "available_tokens": available_tokens,
            "unavailable_tokens": unavailable_tokens,
            "recommendation": available_tokens[0]['symbol'] if available_tokens else "No Alpha tokens available"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/force-alpha-token")
async def force_alpha_token(token_symbol: str = "LIGHTUSDT"):
    """Forzar uso de un token Alpha Events específico"""
    try:
        # Verificar que sea un token Alpha Events válido
        if token_symbol not in bot.alpha_tokens:
            return {
                "status": "error",
                "error": f"{token_symbol} no es un token Alpha Events válido",
                "valid_tokens": list(bot.alpha_tokens.keys())
            }
        
        # Verificar disponibilidad en Binance
        ticker = await bot._make_request('GET', '/api/v3/ticker/24hr', {'symbol': token_symbol})
        exchange_info = await bot._make_request('GET', '/api/v3/exchangeInfo', {'symbol': token_symbol})
        
        if not ticker or not exchange_info:
            return {
                "status": "error",
                "error": f"{token_symbol} no disponible en Binance"
            }
        
        symbol_info = exchange_info['symbols'][0]
        if symbol_info['status'] != 'TRADING':
            return {
                "status": "error",
                "error": f"{token_symbol} no está en estado TRADING (actual: {symbol_info['status']})"
            }
        
        # Intentar ejecutar un trade con este token
        if not bot.should_trade():
            return {
                "status": "error",
                "error": "Sistema no puede operar - límites alcanzados",
                "daily_stats": asdict(bot.daily_stats)
            }
        
        # Ejecutar trade forzado
        metrics = await bot.analyze_token_metrics(token_symbol)
        cycle_volume = min(bot.get_dynamic_volume(), 15)  # Volumen conservador para test
        
        result = await bot.execute_smart_arbitrage(token_symbol, metrics, cycle_volume)
        
        return {
            "status": "success",
            "message": f"Trade ejecutado exitosamente con {token_symbol}",
            "result": result,
            "token_info": bot.alpha_tokens[token_symbol]
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e)
        }
async def test_binance_connection():
    """Test de conexión mejorado con Binance"""
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
        
        # Test de trading permissions
        can_trade = account.get('canTrade', False)
        permissions = account.get('permissions', [])
        
        # Calcular poder de trading
        usdt_balance = balances.get('USDT', 0)
        btc_balance = balances.get('BTC', 0)
        bnb_balance = balances.get('BNB', 0)
        
        # Estimar valor total (precio BTC aproximado)
        try:
            btc_ticker = await bot._make_request('GET', '/api/v3/ticker/price', {'symbol': 'BTCUSDT'})
            btc_price = float(btc_ticker['price'])
            estimated_total = usdt_balance + (btc_balance * btc_price) + (bnb_balance * 600)  # BNB aprox
        except:
            estimated_total = usdt_balance
        
        return {
            "status": "success",
            "server_time": server_time,
            "account_type": account.get('accountType'),
            "can_trade": can_trade,
            "permissions": permissions,
            "balances_with_funds": balances,
            "trading_power": {
                "usdt_available": usdt_balance,
                "estimated_total_value": estimated_total,
                "can_trade_alpha_events": usdt_balance >= MIN_CYCLE_VOLUME
            },
            "alpha_events_readiness": {
                "sufficient_funds": usdt_balance >= DAILY_VOLUME_TARGET * 0.1,
                "trading_enabled": can_trade and 'SPOT' in permissions,
                "recommended_balance": f"${DAILY_VOLUME_TARGET * 0.15:.0f} USDT minimum"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/dashboard")
async def get_dashboard():
    """Dashboard HTML avanzado"""
    progress = (bot.daily_stats.volume / DAILY_VOLUME_TARGET) * 100
    points_progress = (bot.daily_stats.points_earned / TARGET_POINTS) * 100
    
    # Status color
    if bot.should_trade() and bot.daily_stats.loss < 1:
        status_class = "good"
        status_text = "🟢 Sistema Activo - Operando"
    elif bot.should_trade():
        status_class = "warning"
        status_text = "🟡 Sistema Activo - Monitoreando"
    else:
        status_class = "danger"
        status_text = "🔴 Límites Alcanzados"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alpha Events Pro Dashboard</title>
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 20px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 20px;
            }}
            .header h1 {{ 
                color: #333; 
                margin: 0;
                font-size: 2.5em;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stats {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 20px; 
                margin: 30px 0; 
            }}
            .stat {{ 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px; 
                border-radius: 15px; 
                text-align: center;
                border-left: 5px solid #007bff;
                transition: transform 0.3s ease;
            }}
            .stat:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .stat h3 {{ 
                margin: 0 0 15px 0; 
                color: #333; 
                font-size: 1.1em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .stat .value {{ 
                font-size: 2.2em; 
                font-weight: bold; 
                color: #007bff;
                margin-bottom: 5px;
            }}
            .stat .subvalue {{
                font-size: 0.9em;
                color: #666;
            }}
            .progress-container {{
                margin: 30px 0;
            }}
            .progress {{ 
                background: #e9ecef; 
                border-radius: 25px; 
                overflow: hidden; 
                height: 30px; 
                position: relative;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            }}
            .progress-bar {{ 
                background: linear-gradient(90deg, #28a745, #20c997); 
                height: 100%; 
                transition: width 0.5s ease;
                position: relative;
            }}
            .progress-text {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
                color: white;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }}
            .status {{ 
                padding: 20px; 
                border-radius: 15px; 
                margin: 20px 0; 
                text-align: center; 
                font-weight: bold;
                font-size: 1.2em;
            }}
            .status.good {{ 
                background: linear-gradient(135deg, #d4edda, #c3e6cb); 
                color: #155724; 
                border: 2px solid #28a745;
            }}
            .status.warning {{ 
                background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
                color: #856404; 
                border: 2px solid #ffc107;
            }}
            .status.danger {{ 
                background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
                color: #721c24; 
                border: 2px solid #dc3545;
            }}
            .controls {{
                margin-top: 40px; 
                text-align: center;
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 15px;
            }}
            .button {{ 
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white; 
                border: none; 
                padding: 15px 30px; 
                border-radius: 25px; 
                cursor: pointer; 
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,123,255,0.3);
            }}
            .button:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,123,255,0.4);
            }}
            .button.success {{ 
                background: linear-gradient(135deg, #28a745, #20c997);
                box-shadow: 0 4px 15px rgba(40,167,69,0.3);
            }}
            .button.success:hover {{ 
                box-shadow: 0 6px 20px rgba(40,167,69,0.4);
            }}
            .button.warning {{ 
                background: linear-gradient(135deg, #ffc107, #e0a800);
                color: #212529;
                box-shadow: 0 4px 15px rgba(255,193,7,0.3);
            }}
            .button.info {{ 
                background: linear-gradient(135deg, #17a2b8, #138496);
                box-shadow: 0 4px 15px rgba(23,162,184,0.3);
            }}
            .logs {{ 
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                padding: 20px; 
                border-radius: 15px; 
                margin: 30px 0; 
                font-family: 'Courier New', monospace; 
                max-height: 400px; 
                overflow-y: auto; 
                font-size: 14px;
                border: 1px solid #dee2e6;
                display: none;
            }}
            .performance-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .token-performance {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #6c757d;
            }}
            .loading {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Alpha Events Pro Dashboard</h1>
                <p>Sistema Inteligente de Trading Automatizado</p>
                <p>Actualizado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <h3>💰 Volumen Diario</h3>
                    <div class="value">${bot.daily_stats.volume:.2f}</div>
                    <div class="subvalue">Target: ${DAILY_VOLUME_TARGET}</div>
                </div>
                <div class="stat">
                    <h3>⭐ Puntos Alpha</h3>
                    <div class="value">{bot.daily_stats.points_earned}</div>
                    <div class="subvalue">Target: {TARGET_POINTS}</div>
                </div>
                <div class="stat">
                    <h3>📉 Pérdida Diaria</h3>
                    <div class="value">${bot.daily_stats.loss:.4f}</div>
                    <div class="subvalue">Máximo: ${MAX_DAILY_LOSS}</div>
                </div>
                <div class="stat">
                    <h3>🔄 Operaciones</h3>
                    <div class="value">{bot.daily_stats.trades}</div>
                    <div class="subvalue">Completadas hoy</div>
                </div>
                <div class="stat">
                    <h3>🏆 Mejor Token</h3>
                    <div class="value">{bot.daily_stats.best_token or 'N/A'}</div>
                    <div class="subvalue">Más rentable</div>
                </div>
                <div class="stat">
                    <h3>📊 Progreso Total</h3>
                    <div class="value">{max(progress, points_progress):.1f}%</div>
                    <div class="subvalue">Vol/Puntos</div>
                </div>
            </div>
            
            <div class="progress-container">
                <h4>Progreso de Volumen</h4>
                <div class="progress">
                    <div class="progress-bar" style="width: {min(progress, 100)}%">
                        <div class="progress-text">{progress:.1f}% Volume</div>
                    </div>
                </div>
            </div>
            
            <div class="progress-container">
                <h4>Progreso de Puntos Alpha</h4>
                <div class="progress">
                    <div class="progress-bar" style="width: {min(points_progress, 100)}%; background: linear-gradient(90deg, #ffc107, #fd7e14);">
                        <div class="progress-text">{points_progress:.1f}% Points</div>
                    </div>
                </div>
            </div>
            
            <div class="status {status_class}">
                {status_text}
            </div>
            
            <div class="controls">
                <button class="button" onclick="window.location.reload()">
                    🔄 Actualizar Dashboard
                </button>
                <button class="button success" onclick="executeSmartCycle()" id="executeBtn">
                    🎯 Ejecutar Ciclo Inteligente
                </button>
                <button class="button warning" onclick="testConnection()">
                    🔗 Test Conexión Binance
                </button>
                <button class="button info" onclick="analyzeTokens()">
                    📈 Analizar Tokens
                </button>
                <button class="button info" onclick="checkAlphaTokens()">
                    🪙 Verificar Alpha Tokens
                </button>
                <button class="button success" onclick="forceAlphaToken()">
                    ⚡ Forzar Token Alpha
                </button>
            </div>
            
            <div id="logs" class="logs">
                <h4>📋 Log de operaciones:</h4>
                <div id="logContent">Esperando operaciones...</div>
            </div>
            
            <div id="tokenAnalysis" style="display: none; margin-top: 30px;">
                <h3>🔍 Análisis de Tokens Alpha Events</h3>
                <div id="tokenGrid" class="performance-grid"></div>
            </div>
        </div>
        
        <script>
        async function executeSmartCycle() {{
            const btn = document.getElementById('executeBtn');
            const logs = document.getElementById('logs');
            const logContent = document.getElementById('logContent');
            
            btn.innerHTML = '<div class="loading"></div> Ejecutando Ciclo Inteligente...';
            btn.disabled = true;
            logs.style.display = 'block';
            logContent.innerHTML = '🎯 Iniciando análisis inteligente de mercado...';
            
            try {{
                const response = await fetch('/execute-smart-cycle-test');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    const r = result.result;
                    const analysis = result.token_analysis;
                    
                    logContent.innerHTML = `
                        ✅ <strong>Ciclo ejecutado exitosamente</strong><br><br>
                        🪙 <strong>Token seleccionado:</strong> ${{analysis.selected_token}}<br>
                        📊 <strong>Volumen 24h:</strong> ${{analysis.volume_24h}}<br>
                        📈 <strong>Spread:</strong> ${{analysis.spread}}<br>
                        🌊 <strong>Volatilidad:</strong> ${{analysis.volatility}}<br>
                        💧 <strong>Liquidez:</strong> ${{analysis.liquidity_score}}<br>
                        ⭐ <strong>Potencial puntos:</strong> ${{analysis.alpha_points_potential}}<br><br>
                        💰 <strong>Volumen generado:</strong> ${{r.volume_generated?.toFixed(2)}}<br>
                        📊 <strong>P&L:</strong> ${{r.net_pnl?.toFixed(4)}}<br>
                        ⭐ <strong>Puntos estimados:</strong> +${{r.estimated_points}}<br>
                        📈 <strong>Volumen diario:</strong> ${{r.daily_volume?.toFixed(2)}} (${{r.progress_percent?.toFixed(1)}}%)<br>
                        🏆 <strong>Puntos totales:</strong> ${{r.daily_points}}
                    `;
                    setTimeout(() => window.location.reload(), 5000);
                }} else if (result.status === 'stopped') {{
                    logContent.innerHTML = `
                        🛑 <strong>${{result.reason}}</strong><br><br>
                        📊 Estadísticas del día:<br>
                        💰 Volumen: ${{result.daily_stats?.volume?.toFixed(2)}}<br>
                        ⭐ Puntos: ${{result.daily_stats?.points_earned}}<br>
                        🔄 Trades: ${{result.daily_stats?.trades}}<br>
                        📉 Pérdidas: ${{result.daily_stats?.loss?.toFixed(4)}}
                    `;
                }} else {{
                    logContent.innerHTML = `❌ <strong>${{result.status}}:</strong> ${{result.error}}<br><br>💡 ${{result.help || ''}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ <strong>Error de conexión:</strong> ${{error.message}}`;
            }}
            
            btn.innerHTML = '🎯 Ejecutar Ciclo Inteligente';
            btn.disabled = false;
        }}
        
        async function testConnection() {{
            const logContent = document.getElementById('logContent');
            const logs = document.getElementById('logs');
            
            logs.style.display = 'block';
            logContent.innerHTML = '🔗 Verificando conexión con Binance...';
            
            try {{
                const response = await fetch('/test-binance-connection');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    const trading = result.trading_power;
                    const alpha = result.alpha_events_readiness;
                    
                    logContent.innerHTML = `
                        ✅ <strong>Conexión exitosa</strong><br><br>
                        📊 <strong>Información de cuenta:</strong><br>
                        🏦 Tipo: ${{result.account_type}}<br>
                        ✅ Puede operar: ${{result.can_trade}}<br>
                        🔑 Permisos: ${{result.permissions?.join(', ')}}<br><br>
                        💰 <strong>Poder de trading:</strong><br>
                        💵 USDT disponible: ${{trading.usdt_available?.toFixed(2)}}<br>
                        📈 Valor total estimado: ${{trading.estimated_total_value?.toFixed(2)}}<br><br>
                        🎯 <strong>Alpha Events:</strong><br>
                        ✅ Fondos suficientes: ${{alpha.sufficient_funds ? 'Sí' : 'No'}}<br>
                        🔄 Trading habilitado: ${{alpha.trading_enabled ? 'Sí' : 'No'}}<br>
                        💡 Balance recomendado: ${{alpha.recommended_balance}}
                    `;
                }} else {{
                    logContent.innerHTML = `❌ <strong>Error:</strong> ${{result.error}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ <strong>Error:</strong> ${{error.message}}`;
            }}
        }}
        
        async function analyzeTokens() {{
            const logContent = document.getElementById('logContent');
            const logs = document.getElementById('logs');
            const tokenAnalysis = document.getElementById('tokenAnalysis');
            const tokenGrid = document.getElementById('tokenGrid');
            
            logs.style.display = 'block';
            logContent.innerHTML = '📈 Analizando todos los tokens Alpha Events...';
            
            try {{
                const response = await fetch('/analyze-tokens');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    logContent.innerHTML = `
                        ✅ <strong>Análisis completado</strong><br>
                        🪙 Tokens analizados: ${{result.total_tokens}}<br>
                        ⏰ Timestamp: ${{new Date(result.timestamp).toLocaleString()}}
                    `;
                    
                    // Mostrar análisis de tokens
                    tokenAnalysis.style.display = 'block';
                    tokenGrid.innerHTML = '';
                    
                    result.analysis.forEach(token => {{
                        if (!token.error) {{
                            const tokenCard = document.createElement('div');
                            tokenCard.className = 'token-performance';
                            tokenCard.innerHTML = `
                                <h4>${{token.symbol}}</h4>
                                <p><strong>Tier:</strong> ${{token.tier}}</p>
                                <p><strong>Puntos base:</strong> ${{token.base_points}}</p>
                                <p><strong>Potencial:</strong> ⭐${{token.alpha_points_potential}}</p>
                                <p><strong>Volumen 24h:</strong> ${{token.volume_24h?.toLocaleString()}}</p>
                                <p><strong>Volatilidad:</strong> ${{token.volatility?.toFixed(2)}}%</p>
                                <p><strong>Liquidez:</strong> ${{token.liquidity_score?.toFixed(1)}}/100</p>
                            `;
                            tokenGrid.appendChild(tokenCard);
                        }}
                    }});
                }} else {{
                    logContent.innerHTML = `❌ <strong>Error:</strong> ${{result.error}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ <strong>Error:</strong> ${{error.message}}`;
            }}
        async function checkAlphaTokens() {{
            const logContent = document.getElementById('logContent');
            const logs = document.getElementById('logs');
            
            logs.style.display = 'block';
            logContent.innerHTML = '🪙 Verificando tokens Alpha Events en Binance...';
            
            try {{
                const response = await fetch('/check-alpha-tokens');
                const result = await response.json();
                
                if (result.status === 'success') {{
                    let tokenList = '';
                    result.available_tokens.forEach(token => {{
                        const statusEmoji = token.is_trading ? '✅' : '❌';
                        tokenList += `
                            ${{statusEmoji}} <strong>${{token.symbol}}</strong> (Tier ${{token.tier}})<br>
                            📊 Volumen 24h: ${{token.volume_24h?.toLocaleString()}}<br>
                            📈 Cambio: ${{token.price_change?.toFixed(2)}}%<br>
                            ⭐ Puntos base: ${{token.base_points}}<br><br>
                        `;
                    }});
                    
                    logContent.innerHTML = `
                        ✅ <strong>Verificación de Alpha Tokens completada</strong><br><br>
                        📊 <strong>Resumen:</strong><br>
                        🪙 Total tokens Alpha: ${{result.total_alpha_tokens}}<br>
                        ✅ Disponibles: ${{result.available_count}}<br>
                        ❌ No disponibles: ${{result.unavailable_count}}<br>
                        🏆 Recomendado: <strong>${{result.recommendation}}</strong><br><br>
                        <strong>Tokens disponibles:</strong><br>
                        ${{tokenList}}
                    `;
                    
                    if (result.unavailable_tokens.length > 0) {{
                        logContent.innerHTML += `
                            <strong>⚠️ Tokens no disponibles:</strong><br>
                            ${{result.unavailable_tokens.map(t => typeof t === 'string' ? t : `${{t.symbol}}: ${{t.error}}`).join('<br>')}}
                        `;
                    }}
                }} else {{
                    logContent.innerHTML = `❌ <strong>Error:</strong> ${{result.error}}`;
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ <strong>Error:</strong> ${{error.message}}`;
            }}
        }}
        
        async function forceAlphaToken() {{
            const logContent = document.getElementById('logContent');
            const logs = document.getElementById('logs');
            
            // Preguntar qué token usar
            const token = prompt('¿Qué token Alpha Events quieres usar?\\n\\nOpciones:\\n- LIGHTUSDT (Tier 1)\\n- RIVERUSDT (Tier 1)\\n- BLESSUSDT (Tier 1)\\n- HANAUSDT (Tier 2)\\n- COAIUSDT (Tier 2)\\n- ASTERUSDT (Tier 2)\\n- AIXBTUSDT (Tier 3)\\n- MAGICUSDT (Tier 3)\\n- OMNIUSDT (Tier 3)', 'LIGHTUSDT');
            
            if (!token) return;
            
            logs.style.display = 'block';
            logContent.innerHTML = `⚡ Forzando trade con ${{token}}...`;
            
            try {{
                const response = await fetch(`/force-alpha-token?token_symbol=${{token}}`, {{
                    method: 'POST'
                }});
                const result = await response.json();
                
                if (result.status === 'success') {{
                    const r = result.result;
                    logContent.innerHTML = `
                        ✅ <strong>Trade forzado exitoso con ${{token}}</strong><br><br>
                        🪙 <strong>Token:</strong> ${{token}} (Tier ${{result.token_info.tier}})<br>
                        ⭐ <strong>Puntos base:</strong> ${{result.token_info.base_points}}<br><br>
                        💰 <strong>Volumen generado:</strong> ${{r.volume_generated?.toFixed(2)}}<br>
                        📊 <strong>P&L:</strong> ${{r.net_pnl?.toFixed(4)}}<br>
                        ⭐ <strong>Puntos estimados:</strong> +${{r.estimated_points}}<br>
                        📈 <strong>Volumen diario:</strong> ${{r.daily_volume?.toFixed(2)}}<br>
                        🏆 <strong>Puntos totales:</strong> ${{r.daily_points}}
                    `;
                    setTimeout(() => window.location.reload(), 3000);
                }} else {{
                    logContent.innerHTML = `❌ <strong>Error:</strong> ${{result.error}}<br><br>`;
                    if (result.valid_tokens) {{
                        logContent.innerHTML += `💡 <strong>Tokens válidos:</strong> ${{result.valid_tokens.join(', ')}}`;
                    }}
                }}
            }} catch (error) {{
                logContent.innerHTML = `❌ <strong>Error:</strong> ${{error.message}}`;
            }}
        }}
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Tarea automática mejorada
async def intelligent_auto_trading():
    """Sistema de trading automático inteligente"""
    cycle_count = 0
    
    while True:
        try:
            if bot.should_trade():
                current_hour = datetime.utcnow().hour
                
                # Horarios de mayor actividad (mayor frecuencia)
                if 1 <= current_hour <= 8 or 13 <= current_hour <= 16:  # Asia y Europa
                    sleep_time = random.uniform(1800, 3600)  # 30-60 minutos
                else:
                    sleep_time = random.uniform(3600, 7200)  # 1-2 horas
                
                # Selección inteligente y ejecución
                symbol, metrics = await bot.get_optimal_token_advanced()
                cycle_volume = bot.get_dynamic_volume()
                
                # Verificar si vale la pena el trade basado en métricas
                if metrics.liquidity_score > 30 and metrics.alpha_points_potential >= 3:
                    await bot.execute_smart_arbitrage(symbol, metrics, cycle_volume)
                    cycle_count += 1
                    
                    # Notificación cada 5 trades exitosos
                    if cycle_count % 5 == 0:
                        await bot.send_telegram_notification(
                            f"🎯 <b>Auto-Trading Report</b>\n"
                            f"✅ Completed {cycle_count} cycles\n"
                            f"💰 Daily volume: <b>${bot.daily_stats.volume:.2f}</b>\n"
                            f"⭐ Points earned: <b>{bot.daily_stats.points_earned}</b>\n"
                            f"🎪 Progress: <b>{(bot.daily_stats.volume/DAILY_VOLUME_TARGET)*100:.1f}%</b>"
                        )
                else:
                    bot.logger.info(f"Skipping {symbol} - poor metrics (liquidity: {metrics.liquidity_score:.1f}, points: {metrics.alpha_points_potential})")
                    sleep_time = 600  # Reintentar en 10 minutos
                    
            else:
                # Si ya alcanzamos los objetivos, esperar hasta el próximo día
                sleep_time = 3600
                if cycle_count > 0:
                    await bot.send_telegram_notification(
                        f"🏁 <b>Daily Goals Completed!</b>\n"
                        f"🎯 Total cycles: <b>{cycle_count}</b>\n"
                        f"💰 Volume: <b>${bot.daily_stats.volume:.2f}</b>\n"
                        f"⭐ Points: <b>{bot.daily_stats.points_earned}</b>\n"
                        f"🏆 Best token: <b>{bot.daily_stats.best_token}</b>"
                    )
                    cycle_count = 0
                    
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            bot.logger.error(f"Auto-trading error: {str(e)}")
            await bot.send_telegram_notification(f"⚠️ <b>Auto-trading error:</b> {str(e)}")
            await asyncio.sleep(1800)  # Esperar 30 minutos en caso de error

@app.on_event("startup")
async def start_intelligent_trading():
    """Inicia el sistema inteligente de trading automático"""
    bot.logger.info("Starting Alpha Events Pro intelligent auto-trading system")
    await bot.send_telegram_notification(
        f"🚀 <b>Alpha Events Pro Started!</b>\n"
        f"🎯 Daily target: <b>${DAILY_VOLUME_TARGET}</b>\n"
        f"⭐ Points target: <b>{TARGET_POINTS}</b>\n"
        f"💡 Sistema inteligente activado"
    )
    asyncio.create_task(intelligent_auto_trading())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT), metrics, cycle_volume)
        
        # Añadir métricas del token
        result['token_metrics'] = {
            'volume_24h': metrics.volume_24h,
            'spread': metrics.spread,
            'volatility': metrics.volatility,
            'liquidity_score': metrics.liquidity_score,
            'alpha_points_potential': metrics.alpha_points_potential
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/execute-smart-cycle-test")
async def execute_smart_cycle_test():
    """Test endpoint para ciclo inteligente"""
    try:
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            return {
                "status": "error",
                "error": "APIs de Binance no configuradas",
                "help": "Configura BINANCE_API_KEY y BINANCE_SECRET_KEY en Railway"
            }
        
        # Test conexión
        account = await bot._make_request('GET', '/api/v3/account', signed=True)
        
        if not bot.should_trade():
            return {
                "status": "stopped",
                "reason": "Trading limits reached",
                "daily_stats": asdict(bot.daily_stats),
                "api_connection": "OK"
            }
        
        # Ejecutar ciclo inteligente
        symbol, metrics = await bot.get_optimal_token_advanced()
        cycle_volume = bot.get_dynamic_volume()
        
        result = await bot.execute_smart_arbitrage(symbol
