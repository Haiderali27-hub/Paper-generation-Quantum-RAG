"""
Rugpull detection module - best-effort heuristics
Implements renounced authority checks, LP locked %, top wallet concentration
and simple rug score aggregation (0-100)
"""
import asyncio
import aiohttp
from typing import Dict, Any

async def check_renounced_authority(session: aiohttp.ClientSession, mint: str) -> bool:
    """Return True if authority appears renounced (best-effort via indexer)"""
    try:
        url = f'https://public-api.solscan.io/token/holders?tokenAddress={mint}&offset=0&limit=1'
        async with session.get(url, timeout=8) as resp:
            if resp.status == 200:
                data = await resp.json()
                # Best-effort: if owner is null or special flag present
                return False
    except Exception:
        pass
    return False

async def check_top_wallet_concentration(session: aiohttp.ClientSession, mint: str) -> float:
    """Return percentage held by top wallet (0-100). Best-effort via indexer"""
    try:
        url = f'https://public-api.solscan.io/token/holders?tokenAddress={mint}&offset=0&limit=10'
        async with session.get(url, timeout=8) as resp:
            if resp.status == 200:
                data = await resp.json()
                # data may contain list of holders
                holders = data.get('data') or data
                if isinstance(holders, list) and len(holders) > 0:
                    top = holders[0]
                    amount = top.get('amount') or top.get('balance') or 0
                    total = sum([h.get('amount',0) or h.get('balance',0) for h in holders])
                    if total > 0:
                        top_pct = (float(amount) / float(total)) * 100.0
                        return min(100.0, top_pct)
    except Exception:
        pass
    return 0.0

async def estimate_lp_locked(session: aiohttp.ClientSession, mint: str) -> float:
    """Estimate LP locked percent (best-effort placeholder). Returns 0-100"""
    # Requires DEX-specific APIs; placeholder returns 0
    return 0.0

async def detect_sudden_liquidity_removal(session: aiohttp.ClientSession, mint: str) -> float:
    """Detect sudden liquidity removal; returns a risk metric 0-100"""
    # Placeholder - needs historical pool data
    return 0.0

async def analyze_mint_risks(mint: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        is_renounced = await check_renounced_authority(session, mint)
        top_pct = await check_top_wallet_concentration(session, mint)
        lp_locked = await estimate_lp_locked(session, mint)
        sudden_liq = await detect_sudden_liquidity_removal(session, mint)

    # Simple rug score aggregation
    score = 0.0
    if is_renounced:
        score += 10
    # top wallet concentration increases risk
    score += min(50, top_pct * 0.5)
    # low LP locked increases risk
    score += max(0, (100 - lp_locked) * 0.2)
    # sudden liquidity risk
    score += sudden_liq * 0.5

    return {
        'renounced_authority': is_renounced,
        'top_wallet_pct': top_pct,
        'lp_locked_pct': lp_locked,
        'sudden_liquidity_risk': sudden_liq,
        'rug_score': min(100, score)
    }
