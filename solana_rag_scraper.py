"""
Unified Solana Memecoin Scraper with RAG + Qwen Intelligence
============================================================
Combines:
- Token discovery & duplicate prevention
- Full token data fetching (price, volume, liquidity, market_cap, holders)
- Mathematical decision engine (sigmoid-based probability)
- Rug-pull risk analysis
- RAG-powered intelligent analysis using your PDF knowledge base
- Qwen LLM for advanced reasoning and recommendations

Usage:
    python solana_rag_scraper.py --once
    python solana_rag_scraper.py --interval 60 --sources <endpoint1> <endpoint2>
    python solana_rag_scraper.py --analyze <mint_address>  # Analyze specific token
"""

import os
import re
import json
import time
import asyncio
import aiohttp
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# RAG & ML imports
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from langchain_community.vectorstores import FAISS as FAISS_LC
    from langchain_huggingface import HuggingFaceEmbeddings
    import ollama
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("⚠️  RAG dependencies not available. Install: pip install faiss-cpu sentence-transformers langchain-community langchain-huggingface ollama")


# ============================================================================
# CONFIGURATION
# ============================================================================
FAISS_DB_DIR = "faiss_index"
SEEN_MINTS_FILE = "seen_mints.json"
OUTPUT_FILE = "scraped_tokens.json"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QWEN_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Solana API defaults (user should replace with real indexer endpoints)
DEFAULT_SOURCES = [
    # Example: "https://api.solscan.io/v2/token/new",
    # Add your mint discovery endpoints here
]

# Math engine weights
WEIGHTS = {
    'w1': 0.25,  # volume
    'w2': 0.20,  # speed
    'w3': 0.20,  # liquidity
    'w4': 0.15,  # holder growth
    'w5': 0.15,  # anti-rug
    'w6': 0.05   # latency
}

# API settings
API_TIMEOUT = 10
CONCURRENCY_LIMIT = 10


# ============================================================================
# RAG KNOWLEDGE BASE
# ============================================================================
class RAGKnowledge:
    """RAG system to query knowledge base for crypto/trading insights"""
    
    def __init__(self):
        self.embeddings = None
        self.db = None
        self.embed_model = None
        self.available = False
        
        if HAS_RAG and os.path.exists(FAISS_DB_DIR):
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
                self.db = FAISS_LC.load_local(
                    FAISS_DB_DIR, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                self.embed_model = SentenceTransformer(EMB_MODEL)
                self.available = True
                print("✅ RAG knowledge base loaded successfully")
            except Exception as e:
                print(f"⚠️  RAG initialization failed: {e}")
        else:
            print("⚠️  RAG not available (missing dependencies or FAISS index)")
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search knowledge base for relevant context"""
        if not self.available or not query:
            return []
        
        try:
            docs = self.db.similarity_search(query, k=k)
            return [doc.page_content[:500] for doc in docs]
        except Exception as e:
            print(f"⚠️  RAG search error: {e}")
            return []
    
    def get_trading_insights(self, token_data: Dict) -> List[str]:
        """Get trading strategy insights from knowledge base"""
        queries = [
            "cryptocurrency trading risk assessment",
            "memecoin liquidity analysis",
            "token holder distribution patterns",
            "rug pull detection indicators"
        ]
        
        all_insights = []
        for q in queries:
            results = self.search(q, k=2)
            all_insights.extend(results)
        
        return all_insights[:4]  # Top 4 most relevant


# ============================================================================
# QWEN AI ANALYSIS
# ============================================================================
class QwenAnalyzer:
    """Use Qwen LLM for intelligent token analysis and recommendations"""
    
    def __init__(self, model: str = QWEN_MODEL):
        self.model = model
        self.available = False
        
        if HAS_RAG:
            try:
                # Test if Qwen is available
                ollama.list()
                self.available = True
                print(f"✅ Qwen model '{model}' ready")
            except Exception as e:
                print(f"⚠️  Qwen not available: {e}")
    
    def analyze_token(self, token_data: Dict, rag_context: List[str] = None) -> str:
        """Get AI analysis of token with RAG context"""
        if not self.available:
            return "AI analysis unavailable"
        
        # Build prompt with token data and RAG context
        context_text = "\n\n".join(rag_context) if rag_context else "No additional context available"
        
        prompt = f"""You are a crypto trading expert analyzing a Solana memecoin. 

TOKEN DATA:
- Symbol: {token_data.get('symbol', 'N/A')}
- Price: ${token_data.get('price', 0):.8f}
- Market Cap: ${token_data.get('market_cap', 0):,.2f}
- Liquidity: ${token_data.get('liquidity', 0):,.2f}
- Volume (24h): ${token_data.get('volume', 0):,.2f}
- Holders: {token_data.get('holders', 0)}
- Risk Score: {token_data.get('risk_score', 50)}/100
- Profit Probability: {token_data.get('profit_probability', 0.5):.2%}
- Age: {token_data.get('age_days', 0)} days

RUG PULL ANALYSIS:
{json.dumps(token_data.get('rug_details', {}), indent=2)}

KNOWLEDGE BASE CONTEXT:
{context_text}

Provide a concise analysis (3-4 sentences) covering:
1. Overall risk assessment
2. Key red/green flags
3. Trading recommendation (BUY/HOLD/AVOID)
4. Suggested action

Keep it brief and actionable."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )
            return response['message']['content']
        except Exception as e:
            return f"AI analysis error: {str(e)[:100]}"


# ============================================================================
# DUPLICATE PREVENTION (FETCHER)
# ============================================================================
class Fetcher:
    """Discover new mints and prevent duplicates"""
    
    def __init__(self, sources: List[str]):
        self.sources = sources or DEFAULT_SOURCES
        self.seen = set()
        self._load_seen()
    
    def _load_seen(self):
        if os.path.exists(SEEN_MINTS_FILE):
            try:
                with open(SEEN_MINTS_FILE, 'r') as f:
                    data = json.load(f)
                    self.seen = set(data.get('seen', []))
            except:
                pass
    
    def _save_seen(self):
        with open(SEEN_MINTS_FILE, 'w') as f:
            json.dump({'seen': list(self.seen)}, f)
    
    async def fetch_from_source(self, session: aiohttp.ClientSession, source: str) -> List[str]:
        try:
            async with session.get(source, timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list):
                        return [item.get('mint') for item in data if item.get('mint')]
                    return []
        except:
            return []
        return []
    
    async def discover_mints(self, batch_sources: List[str] = None) -> List[Dict]:
        sources = batch_sources or self.sources
        if not sources:
            return []
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_from_source(session, s) for s in sources]
            results = await asyncio.gather(*tasks)
        
        all_mints = []
        for mints in results:
            all_mints.extend(mints)
        
        new_mints = []
        for mint in all_mints:
            if mint and mint not in self.seen:
                self.seen.add(mint)
                new_mints.append({'mint': mint})
        
        if new_mints:
            self._save_seen()
        
        return new_mints


# ============================================================================
# DATA FETCHER
# ============================================================================
async def fetch_token_info(session: aiohttp.ClientSession, mint: str) -> Dict:
    """Fetch full token metadata and market data"""
    token_info = {
        'mint': mint,
        'name': None,
        'symbol': None,
        'price': 0.0,
        'volume': 0.0,
        'liquidity': 0.0,
        'market_cap': 0.0,
        'holders': 0,
        'dex_buys': 0,
        'dex_sells': 0,
        'age_days': 0,
        'metadata': {}
    }
    
    # Solscan token metadata
    try:
        url = f"https://api.solscan.io/token/meta?token={mint}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as resp:
            if resp.status == 200:
                data = await resp.json()
                token_info['name'] = data.get('name')
                token_info['symbol'] = data.get('symbol')
                token_info['metadata'] = data
    except:
        pass
    
    # CoinGecko contract data for price/volume
    try:
        url = f"https://api.coingecko.com/api/v3/coins/solana/contract/{mint}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as resp:
            if resp.status == 200:
                data = await resp.json()
                md = data.get('market_data', {})
                token_info['price'] = md.get('current_price', {}).get('usd', 0.0)
                token_info['volume'] = md.get('total_volume', {}).get('usd', 0.0)
                token_info['market_cap'] = md.get('market_cap', {}).get('usd', 0.0)
    except:
        pass
    
    # Holder count from Solscan
    try:
        url = f"https://api.solscan.io/token/holders?token={mint}&offset=0&size=1"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as resp:
            if resp.status == 200:
                data = await resp.json()
                token_info['holders'] = data.get('total', 0)
    except:
        pass
    
    return token_info


async def fetch_batch_token_info(mints: List[str], concurrency: int = CONCURRENCY_LIMIT) -> List[Dict]:
    """Fetch token info in parallel with concurrency control"""
    sem = asyncio.Semaphore(concurrency)
    
    async def fetch_with_sem(session, mint):
        async with sem:
            return await fetch_token_info(session, mint)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_sem(session, m) for m in mints]
        return await asyncio.gather(*tasks)


# ============================================================================
# MATH ENGINE
# ============================================================================
def sigmoid(x: float) -> float:
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-x))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize to [0,1]"""
    if max_val <= min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def latency_score(latency_seconds: float) -> float:
    """Score based on response latency (lower is better)"""
    if latency_seconds <= 0.1:
        return 1.0
    elif latency_seconds <= 1.0:
        return 0.8
    elif latency_seconds <= 3.0:
        return 0.5
    return 0.2


def compute_speed_score(metrics: Dict) -> float:
    """Compute speed score from volume/holder change rates"""
    vol_change = metrics.get('volume_change_pct', 0.0)
    holder_change = metrics.get('holders_change_pct', 0.0)
    buys_per_min = metrics.get('buys_per_min', 0.0)
    
    vol_norm = normalize(vol_change, 0, 100)
    holder_norm = normalize(holder_change, 0, 50)
    buys_norm = normalize(buys_per_min, 0, 10)
    
    return (vol_norm * 0.4 + holder_norm * 0.3 + buys_norm * 0.3)


def compute_probability(metrics: Dict, weights: Dict = WEIGHTS) -> float:
    """Compute profit probability using weighted sigmoid"""
    w1 = weights['w1']
    w2 = weights['w2']
    w3 = weights['w3']
    w4 = weights['w4']
    w5 = weights['w5']
    w6 = weights['w6']
    
    volume_score = metrics.get('volume_score', 0.0)
    speed_score = metrics.get('speed_score', 0.0)
    liquidity_score = metrics.get('liquidity_score', 0.0)
    holder_growth = metrics.get('holder_growth_score', 0.0)
    anti_rug = metrics.get('anti_rug_score', 0.5)
    latency_s = metrics.get('latency_seconds', 1.0)
    
    lat_score = latency_score(latency_s)
    
    weighted_sum = (w1 * volume_score + 
                    w2 * speed_score + 
                    w3 * liquidity_score + 
                    w4 * holder_growth + 
                    w5 * anti_rug + 
                    w6 * lat_score)
    
    # Sigmoid transform
    z = (weighted_sum - 0.5) * 10  # Center and scale
    probability = sigmoid(z)
    return probability


def final_score_combination(probability: float, rug_score: float) -> float:
    """Combine probability and rug risk for final score"""
    anti_rug_factor = (100 - rug_score) / 100.0
    final = probability * 0.7 + anti_rug_factor * 0.3
    return final


# ============================================================================
# RUG PULL DETECTION
# ============================================================================
async def analyze_mint_risks(mint: str) -> Dict[str, Any]:
    """Best-effort rug-pull heuristics"""
    risks = {
        'rug_score': 50,  # 0=safe, 100=high risk
        'renounced_authority': False,
        'top_wallet_concentration': 0.0,
        'lp_locked': False,
        'lp_locked_pct': 0.0,
        'sudden_liquidity_drop': False,
        'details': []
    }
    
    # Check renounced authority via Solscan
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.solscan.io/token/meta?token={mint}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    authority = data.get('authority')
                    if not authority or authority == "null":
                        risks['renounced_authority'] = True
                        risks['rug_score'] -= 15
    except:
        pass
    
    # Check top wallet concentration
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.solscan.io/token/holders?token={mint}&offset=0&size=10"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    holders = data.get('data', [])
                    if holders:
                        top_pct = holders[0].get('amount_percent', 0.0)
                        risks['top_wallet_concentration'] = top_pct
                        if top_pct > 50:
                            risks['rug_score'] += 30
                            risks['details'].append(f"Top wallet owns {top_pct:.1f}%")
                        elif top_pct > 30:
                            risks['rug_score'] += 15
    except:
        pass
    
    # LP lock estimation (placeholder - requires DEX-specific API)
    risks['lp_locked'] = False  # Would need DEX pool analysis
    
    # Ensure rug_score stays in [0, 100]
    risks['rug_score'] = max(0, min(100, risks['rug_score']))
    
    return risks


# ============================================================================
# ORCHESTRATOR
# ============================================================================
class SolanaRAGOrchestrator:
    """Main orchestrator with RAG + Qwen intelligence"""
    
    def __init__(self, sources: List[str] = None):
        self.fetcher = Fetcher(sources or DEFAULT_SOURCES)
        self.rag = RAGKnowledge()
        self.qwen = QwenAnalyzer()
    
    async def process_new_mints(self, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Discover, analyze, and score new mints with AI insights"""
        new = await self.fetcher.discover_mints(batch_sources=sources)
        if not new:
            return []
        
        mints = [item['mint'] for item in new]
        print(f"🔍 Found {len(mints)} new mints, fetching data...")
        
        # Fetch batch token info
        token_infos = await fetch_batch_token_info(mints)
        
        results = []
        for token in token_infos:
            mint = token.get('mint')
            
            # Compute metrics for math engine
            metrics = {
                'volume_score': 0.0,
                'speed_score': 0.0,
                'liquidity_score': 0.0,
                'holder_growth_score': 0.0,
                'latency_seconds': 0.01
            }
            
            vol = token.get('volume') or 0.0
            metrics['volume_score'] = 0.0 if vol <= 0 else min(1.0, (vol ** 0.5) / 1000.0)
            metrics['liquidity_score'] = 0.0 if token.get('liquidity', 0) <= 0 else 1.0
            metrics['holder_growth_score'] = min(1.0, token.get('holders', 0) / 1000.0)
            
            # Rug analysis
            risks = await analyze_mint_risks(mint)
            rug_score = risks.get('rug_score', 50)
            metrics['anti_rug_score'] = (100 - rug_score) / 100.0
            
            # Compute probability
            probability = compute_probability(metrics)
            final = final_score_combination(probability, rug_score)
            
            # RAG insights
            rag_context = []
            if self.rag.available:
                rag_context = self.rag.get_trading_insights(token)
            
            # Build result object
            result = {
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
                'rug_details': risks,
                'rag_insights': rag_context[:2] if rag_context else [],  # Top 2 insights
                'ai_analysis': None
            }
            
            # Qwen AI analysis
            if self.qwen.available:
                result['ai_analysis'] = self.qwen.analyze_token(result, rag_context)
            
            results.append(result)
        
        return results
    
    async def analyze_specific_token(self, mint: str) -> Dict[str, Any]:
        """Deep analysis of a specific token with full AI insights"""
        print(f"🔬 Analyzing token: {mint}")
        
        # Fetch token data
        async with aiohttp.ClientSession() as session:
            token = await fetch_token_info(session, mint)
        
        # Metrics
        metrics = {
            'volume_score': min(1.0, (token.get('volume', 0) ** 0.5) / 1000.0),
            'speed_score': 0.5,  # Placeholder
            'liquidity_score': 1.0 if token.get('liquidity', 0) > 0 else 0.0,
            'holder_growth_score': min(1.0, token.get('holders', 0) / 1000.0),
            'latency_seconds': 0.01
        }
        
        # Rug analysis
        risks = await analyze_mint_risks(mint)
        rug_score = risks.get('rug_score', 50)
        metrics['anti_rug_score'] = (100 - rug_score) / 100.0
        
        # Probability
        probability = compute_probability(metrics)
        final = final_score_combination(probability, rug_score)
        
        # RAG insights
        rag_context = []
        if self.rag.available:
            queries = [
                f"token analysis {token.get('symbol', '')}",
                "cryptocurrency risk assessment",
                "memecoin trading strategy"
            ]
            for q in queries:
                rag_context.extend(self.rag.search(q, k=2))
        
        result = {
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
            'rug_details': risks,
            'rag_insights': rag_context[:3],
            'ai_analysis': None
        }
        
        # Qwen analysis
        if self.qwen.available:
            result['ai_analysis'] = self.qwen.analyze_token(result, rag_context)
        
        return result


# ============================================================================
# CLI & MAIN
# ============================================================================
async def run_once(output_file: str = None, sources: List[str] = None):
    """Run one iteration"""
    orch = SolanaRAGOrchestrator(sources)
    results = await orch.process_new_mints(sources)
    
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
    
    print(json.dumps(results, indent=2))
    print(f"\n✅ Processed {len(results)} tokens")


async def run_loop(interval: int, output_file: str = None, sources: List[str] = None):
    """Continuous polling"""
    orch = SolanaRAGOrchestrator(sources)
    
    while True:
        results = await orch.process_new_mints(sources)
        ts = datetime.utcnow().isoformat() + 'Z'
        
        if results:
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r) + '\n')
            print(f"[{ts}] Found {len(results)} new mints")
            for r in results:
                print(f"  • {r['symbol']} | Risk: {r['risk_score']}/100 | Prob: {r['profit_probability']:.2%}")
                if r.get('ai_analysis'):
                    print(f"    AI: {r['ai_analysis'][:100]}...")
        else:
            print(f"[{ts}] No new mints found")
        
        await asyncio.sleep(interval)


async def analyze_token(mint: str):
    """Analyze specific token"""
    orch = SolanaRAGOrchestrator()
    result = await orch.analyze_specific_token(mint)
    
    print("\n" + "="*80)
    print(f"TOKEN ANALYSIS: {result['symbol']} ({mint})")
    print("="*80)
    print(f"Price:          ${result['price']:.8f}")
    print(f"Market Cap:     ${result['market_cap']:,.2f}")
    print(f"Volume (24h):   ${result['volume']:,.2f}")
    print(f"Liquidity:      ${result['liquidity']:,.2f}")
    print(f"Holders:        {result['holders']}")
    print(f"\nRisk Score:     {result['risk_score']}/100")
    print(f"Probability:    {result['profit_probability']:.2%}")
    print(f"Final Score:    {result['final_score']:.4f}")
    
    print(f"\nRUG ANALYSIS:")
    for key, val in result['rug_details'].items():
        print(f"  {key}: {val}")
    
    if result.get('rag_insights'):
        print(f"\nKNOWLEDGE BASE INSIGHTS:")
        for i, insight in enumerate(result['rag_insights'], 1):
            print(f"\n{i}. {insight[:300]}...")
    
    if result.get('ai_analysis'):
        print(f"\nQWEN AI ANALYSIS:")
        print(result['ai_analysis'])
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Solana Memecoin Scraper with RAG + Qwen Intelligence')
    parser.add_argument('--once', action='store_true', help='Run one iteration and exit')
    parser.add_argument('--interval', type=int, default=0, help='Poll interval in seconds')
    parser.add_argument('--out', type=str, default=OUTPUT_FILE, help='Output file for JSON lines')
    parser.add_argument('--sources', type=str, nargs='*', help='Mint discovery endpoints')
    parser.add_argument('--analyze', type=str, help='Analyze specific token by mint address')
    
    args = parser.parse_args()
    
    if args.analyze:
        asyncio.run(analyze_token(args.analyze))
    elif args.once or args.interval == 0:
        asyncio.run(run_once(output_file=args.out, sources=args.sources))
    else:
        asyncio.run(run_loop(interval=args.interval, output_file=args.out, sources=args.sources))


if __name__ == '__main__':
    print("🚀 Solana RAG Scraper Starting...")
    print(f"📚 RAG Available: {HAS_RAG}")
    print(f"🤖 Qwen Model: {QWEN_MODEL}")
    print("="*80)
    main()
