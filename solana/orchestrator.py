"""
Main orchestrator: ties fetcher, data, math_engine, rugpull modules
Produces structured JSON output per mint and supports async batch processing
"""
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from fetcher import Fetcher
from data import fetch_batch_token_info
from math_engine import compute_speed_score, compute_probability, latency_score, final_score_combination
from rugpull import analyze_mint_risks

DEFAULT_SOURCES = [
    # User can replace or add provider endpoints returning recently minted tokens
    # For demo, these endpoints are placeholders and should return JSON list [{"mint": "..."}, ...]
]

class Orchestrator:
    def __init__(self, sources: List[str]=None):
        self.fetcher = Fetcher(sources or DEFAULT_SOURCES)

    async def process_new_mints(self, sources: List[str]=None, pre_fetched: List[Dict]=None) -> List[Dict[str, Any]]:
        """
        Process new mints from sources or pre-fetched list
        
        Args:
            sources: List of API endpoints to fetch from
            pre_fetched: Pre-fetched mint data (from pump.fun WebSocket, etc)
        """
        if pre_fetched:
            # Use pre-fetched tokens (from pump.fun)
            new = pre_fetched
        else:
            # Fetch from sources
            new = await self.fetcher.discover_mints(batch_sources=sources)
        
        if not new:
            return []

        mints = [item['mint'] for item in new]
        # Fetch batch token info
        token_infos = await fetch_batch_token_info(mints)

        results = []
        # For each token, run rug analysis and math engine
        async def process(token):
            mint = token.get('mint')
            # Collect quick metrics for math engine
            metrics = {
                'volume_score': 0.0,
                'speed_score': 0.0,
                'liquidity_score': 0.0,
                'holder_growth_score': 0.0,
                'latency_seconds': 0.01
            }
            # compute speed score from available token fields
            metrics['speed_score'] = compute_speed_score({
                'volume_change_pct': token.get('volume_change_pct', 0.0),
                'holders_change_pct': token.get('holders_change_pct', 0.0),
                'buys_per_min': token.get('buys_per_min', 0.0)
            })
            # simple volume score normalization
            vol = token.get('volume') or 0.0
            metrics['volume_score'] = 0.0 if vol <= 0 else min(1.0, (vol ** 0.5) / 1000.0)
            metrics['liquidity_score'] = 0.0 if token.get('liquidity',0) <= 0 else 1.0
            metrics['holder_growth_score'] = min(1.0, token.get('holders',0)/1000.0)

            # rug analysis
            risks = await analyze_mint_risks(mint)
            rug_score = risks.get('rug_score', 50)

            probability = compute_probability({
                'volume_score': metrics['volume_score'],
                'speed_score': metrics['speed_score'],
                'liquidity_score': metrics['liquidity_score'],
                'holder_growth_score': metrics['holder_growth_score'],
                'anti_rug_score': (100 - rug_score)/100.0,
                'latency_seconds': metrics['latency_seconds']
            })

            final = final_score_combination(probability, rug_score)

            out = {
                'mint': mint,
                'name': token.get('name'),
                'symbol': token.get('symbol'),
                'price': token.get('price'),
                'volume': token.get('volume'),
                'liquidity': token.get('liquidity'),
                'market_cap': token.get('market_cap'),
                'holders': token.get('holders'),
                'risk_score': rug_score,
                'profit_probability': round(probability, 4),
                'final_score': round(final, 4),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'metadata': token.get('metadata', {}),
                'rug_details': risks
            }
            results.append(out)

        tasks = [process(t) for t in token_infos]
        await asyncio.gather(*tasks)
        return results

async def run_once(sources: List[str]=None):
    orch = Orchestrator(sources)
    res = await orch.process_new_mints(sources)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    asyncio.run(run_once())
