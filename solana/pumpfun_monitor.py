"""
Pump.fun Real-Time Token Monitor
Connects to pump.fun WebSocket to get NEWLY CREATED tokens in real-time
100% FREE - no API key needed
"""

import asyncio
import json
import websockets
from typing import List, Dict, Any
from datetime import datetime

class PumpFunMonitor:
    """Monitor pump.fun for newly created tokens via WebSocket"""
    
    def __init__(self):
        # Pump.fun public WebSocket endpoint
        self.ws_url = "wss://pumpportal.fun/api/data"
        self.new_tokens = []
        self.seen_tokens = set()
        
    async def connect_and_monitor(self, callback=None, duration_seconds=60):
        """
        Connect to pump.fun WebSocket and monitor for new token creations
        
        Args:
            callback: Optional async function to call with each new token
            duration_seconds: How long to monitor (default 60s)
        """
        try:
            async with websockets.connect(self.ws_url) as ws:
                print(f"🔗 Connected to pump.fun WebSocket")
                
                # Subscribe to new token events
                subscribe_msg = {
                    "method": "subscribeNewToken"
                }
                await ws.send(json.dumps(subscribe_msg))
                print("📡 Subscribed to new token creations")
                
                start_time = asyncio.get_event_loop().time()
                
                while True:
                    # Check timeout
                    if duration_seconds and (asyncio.get_event_loop().time() - start_time) > duration_seconds:
                        print(f"⏱️  Monitoring period ended ({duration_seconds}s)")
                        break
                    
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        # Extract token info
                        if self._is_new_token_event(data):
                            token_info = self._extract_token_info(data)
                            
                            if token_info and token_info['mint'] not in self.seen_tokens:
                                self.seen_tokens.add(token_info['mint'])
                                self.new_tokens.append(token_info)
                                
                                print(f"🆕 NEW TOKEN: {token_info['name']} ({token_info['symbol']}) - {token_info['mint'][:12]}...")
                                
                                if callback:
                                    await callback(token_info)
                    
                    except asyncio.TimeoutError:
                        # No message received, continue waiting
                        continue
                    except Exception as e:
                        print(f"⚠️  Error processing message: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            return []
        
        return self.new_tokens
    
    def _is_new_token_event(self, data: Dict) -> bool:
        """Check if message is a new token creation event"""
        if not isinstance(data, dict):
            return False
        
        # Pump.fun sends different event types
        # Look for token creation indicators
        event_type = data.get('type') or data.get('event') or data.get('txType')
        
        return event_type in ['create', 'newToken', 'tokenCreated', 'mint']
    
    def _extract_token_info(self, data: Dict) -> Dict[str, Any]:
        """Extract token information from WebSocket message"""
        try:
            # Pump.fun message structure varies, try multiple patterns
            token = {}
            
            # Pattern 1: Direct fields
            if 'mint' in data:
                token['mint'] = data['mint']
            elif 'tokenAddress' in data:
                token['mint'] = data['tokenAddress']
            elif 'address' in data:
                token['mint'] = data['address']
            else:
                return None
            
            # Name and symbol
            token['name'] = data.get('name') or data.get('tokenName') or 'Unknown'
            token['symbol'] = data.get('symbol') or data.get('ticker') or 'UNKNOWN'
            
            # Optional fields
            token['creator'] = data.get('creator') or data.get('traderPublicKey')
            token['timestamp'] = data.get('timestamp') or datetime.now().isoformat()
            token['signature'] = data.get('signature') or data.get('txHash')
            
            # Initial data (may be None/0 for new tokens)
            token['initialBuy'] = data.get('initialBuy') or 0
            token['marketCapSol'] = data.get('marketCapSol') or 0
            
            return token
            
        except Exception as e:
            print(f"⚠️  Error extracting token info: {e}")
            return None
    
    def get_discovered_tokens(self) -> List[Dict[str, Any]]:
        """Get all tokens discovered during monitoring"""
        return self.new_tokens


# Standalone monitoring function
async def monitor_pumpfun(duration_seconds=60) -> List[Dict[str, Any]]:
    """
    Monitor pump.fun for new tokens for specified duration
    
    Args:
        duration_seconds: How long to monitor (default 60s)
        
    Returns:
        List of newly created tokens
    """
    monitor = PumpFunMonitor()
    tokens = await monitor.connect_and_monitor(duration_seconds=duration_seconds)
    return tokens


# Test function
async def test_pumpfun():
    """Test pump.fun monitoring for 60 seconds"""
    print("=" * 80)
    print("🚀 PUMP.FUN REAL-TIME MONITOR TEST")
    print("=" * 80)
    print("Monitoring for 60 seconds...\n")
    
    tokens = await monitor_pumpfun(duration_seconds=60)
    
    print("\n" + "=" * 80)
    print(f"📊 RESULTS: Found {len(tokens)} new tokens")
    print("=" * 80)
    
    for i, token in enumerate(tokens, 1):
        print(f"\n{i}. {token['name']} ({token['symbol']})")
        print(f"   Mint: {token['mint']}")
        print(f"   Creator: {token.get('creator', 'Unknown')[:12]}...")
        print(f"   Time: {token.get('timestamp')}")


if __name__ == '__main__':
    # Run test
    asyncio.run(test_pumpfun())
