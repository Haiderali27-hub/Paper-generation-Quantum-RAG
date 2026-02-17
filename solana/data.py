"""
Data module: fetches token metadata, price, volume, liquidity, holders
Uses DexScreener API (FREE) for real-time market data + Solscan for metadata
"""
import asyncio
import aiohttp
from typing import Dict, Any, Optional

# DexScreener proper API - FREE, returns full market data
DEXSCREENER_TOKEN_API = 'https://api.dexscreener.com/latest/dex/tokens/'

async def fetch_token_info(session: aiohttp.ClientSession, mint: str) -> Dict[str, Any]:
    """Fetch contract metadata and basic metrics for a mint."""
    result = {
        'mint': mint,
        'name': None,
        'symbol': None,
        'price': None,
        'volume': 0,
        'liquidity': 0,
        'market_cap': None,
        'holders': 0,
        'dex_buys': 0,
        'dex_sells': 0,
        'age_days': None,
        'metadata': {}
    }
    
    # PRIMARY: DexScreener proper API - FREE and provides full market data
    try:
        url = f'{DEXSCREENER_TOKEN_API}{mint}'
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                pairs = data.get('pairs', [])
                if pairs:
                    # Use the pair with highest liquidity (most reliable data)
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    
                    # Extract market data
                    result['price'] = float(best_pair.get('priceUsd', 0) or 0)
                    result['volume'] = float(best_pair.get('volume', {}).get('h24', 0) or 0)
                    result['liquidity'] = float(best_pair.get('liquidity', {}).get('usd', 0) or 0)
                    
                    # Extract token info
                    base_token = best_pair.get('baseToken', {})
                    result['name'] = base_token.get('name') or result['name']
                    result['symbol'] = base_token.get('symbol') or result['symbol']
                    
                    # Calculate market cap if we have price and supply
                    if result['price'] and result['liquidity']:
                        # Rough estimate: market_cap ≈ liquidity * 10 (typical for new tokens)
                        result['market_cap'] = result['liquidity'] * 10
                    
                    # Store pair info in metadata
                    result['metadata']['dex'] = best_pair.get('dexId', 'unknown')
                    result['metadata']['pair_address'] = best_pair.get('pairAddress', '')
                    result['metadata']['price_change_24h'] = best_pair.get('priceChange', {}).get('h24', 0)
                    result['metadata']['pair_created_at'] = best_pair.get('pairCreatedAt', 0)
                    
                    # Calculate token age in hours
                    if result['metadata']['pair_created_at']:
                        import time
                        age_hours = (time.time() * 1000 - result['metadata']['pair_created_at']) / (1000 * 3600)
                        result['age_days'] = age_hours / 24
    except Exception as e:
        # If DexScreener fails, try fallback sources
        pass

    # FALLBACK: Try Solscan for token metadata if still missing
    if not result['name'] or not result['symbol']:
        try:
            url = f'https://public-api.solscan.io/token/meta?tokenAddress={mint}'
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result['name'] = data.get('name') or result['name']
                    result['symbol'] = data.get('symbol') or result['symbol']
        except Exception:
            pass

    # Placeholder: holders and liquidity - these require chain queries or indexer endpoints
    # Use Solana indexers where available
    try:
        # Example Solscan holders endpoint
        url3 = f'https://public-api.solscan.io/token/holders?tokenAddress={mint}&offset=0&limit=1'
        async with session.get(url3, timeout=10) as resp3:
            if resp3.status == 200:
                d3 = await resp3.json()
                # Solscan returns total count in some field - best-effort
                total = d3.get('total')
                if total is not None:
                    result['holders'] = int(total)
    except Exception:
        pass

    # Liquidity estimation - best-effort via DEX pools (placeholder set to 0)
    result['liquidity'] = result.get('liquidity', 0) or 0

    return result

async def fetch_batch_token_info(mints, concurrency=8):
    results = []
    sem = asyncio.Semaphore(concurrency)
    try:
        async with aiohttp.ClientSession() as session:
            async def _fetch(m):
                async with sem:
                    return await fetch_token_info(session, m)
            tasks = [asyncio.create_task(_fetch(m)) for m in mints]
            for t in asyncio.as_completed(tasks):
                try:
                    r = await t
                    results.append(r)
                except asyncio.CancelledError:
                    pass  # Gracefully handle cancellation
                except Exception:
                    pass
    except asyncio.CancelledError:
        pass  # Gracefully handle cancellation
    return results
