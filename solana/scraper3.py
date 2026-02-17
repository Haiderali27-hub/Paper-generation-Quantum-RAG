"""
🚀 SOLANA MEMECOIN SCRAPER WITH RAG + QWEN AI
Continuously scrapes NEW tokens from multiple sources:
- Dexscreener (new pairs)
- Pump.fun (new launches)
- Birdeye (trending/new)
- GMGN.ai (new listings)
- Photon/BullX feeds

Usage:
  python scraper3.py              # Run continuously (default 60s interval)
  python scraper3.py --interval 30  # Custom interval
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Import from local modules (not package)
from orchestrator import Orchestrator
from pumpfun_monitor import PumpFunMonitor

# ============================================================================
# RAG + QWEN AI INTEGRATION
# ============================================================================

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    import ollama
    HAS_AI = True
except ImportError:
    HAS_AI = False

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_DB = os.path.join(WORKSPACE_ROOT, "faiss_index")
# Use qwen2.5:7b (4.7 GB) instead of qwen3-vl:32b (needs 24.7 GB)
# Falls back to qwen2.5:1.5b if memory issues persist
QWEN_MODEL = "qwen2.5:7b"
QWEN_FALLBACK = "qwen2.5:1.5b"

class RAGEngine:
    def __init__(self):
        self.available = False
        self.db = None
        if HAS_AI and os.path.exists(FAISS_DB):
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.db = FAISS.load_local(FAISS_DB, embeddings, allow_dangerous_deserialization=True)
                self.available = True
                print("✅ RAG Knowledge Base loaded")
            except Exception as e:
                print(f"⚠️  RAG not available: {e}")
    
    def search(self, query: str, k: int = 2) -> List[str]:
        if not self.available:
            return []
        try:
            docs = self.db.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except:
            return []

class QwenAI:
    def __init__(self):
        self.available = False
        self.client = None
        self.model = QWEN_MODEL
        if HAS_AI:
            try:
                self.client = ollama
                # Try primary model (7b), fallback to smaller if it fails
                try:
                    self.client.show(QWEN_MODEL)
                    self.model = QWEN_MODEL
                    print(f"✅ Qwen '{QWEN_MODEL}' ready (4.7 GB)")
                except:
                    # Fallback to smaller model
                    self.client.show(QWEN_FALLBACK)
                    self.model = QWEN_FALLBACK
                    print(f"✅ Qwen '{QWEN_FALLBACK}' ready (fallback, 986 MB)")
                self.available = True
            except Exception as e:
                print(f"⚠️  Qwen not available: {e}")
    
    def analyze(self, token_data: Dict, rag_insights: List[str]) -> str:
        if not self.available:
            return None
        
        # Handle None values for formatting
        price = token_data.get('price') or 0.0
        volume = token_data.get('volume') or 0.0
        market_cap = token_data.get('market_cap') or 0.0
        holders = token_data.get('holders') or 0
        
        prompt = f"""Analyze this Solana token and provide BUY/HOLD/AVOID recommendation:

Token: {token_data.get('symbol', 'Unknown')} ({token_data.get('name', 'Unknown')})
Price: ${price:.8f}
Volume 24h: ${volume:,.0f}
Market Cap: ${market_cap:,.0f}
Holders: {holders}
Risk Score: {token_data.get('risk_score', 50)}/100
Profit Probability: {token_data.get('profit_probability', 0)*100:.2f}%

Rug Analysis: {token_data.get('rug_details', {})}

Knowledge Base Context:
{rag_insights[0] if rag_insights else 'No context available'}

Provide: 1) BUY/HOLD/AVOID decision, 2) Key reasons (2-3 bullet points), 3) Risk level"""

        try:
            response = self.client.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
        except Exception as e:
            return f"AI error: {str(e)[:100]}"

# ============================================================================
# TOKEN DISCOVERY SOURCES (NEW COINS)
# ============================================================================

# Using DexScreener's working endpoints + Pump.fun WebSocket
DEFAULT_SOURCES = [
    # DexScreener - Featured/promoted tokens
    "https://api.dexscreener.com/token-profiles/latest/v1",
    "https://api.dexscreener.com/token-boosts/latest/v1",
]

# Pump.fun WebSocket for REAL new tokens (enabled by default)
USE_PUMPFUN_WEBSOCKET = True

# ============================================================================
# ENHANCED ORCHESTRATOR WITH AI
# ============================================================================

class AIOrchestrator(Orchestrator):
    def __init__(self, sources: List[str] = None):
        super().__init__(sources)
        self.rag = RAGEngine()
        self.qwen = QwenAI()
    
    async def process_new_mints(self, sources: List[str] = None) -> List[Dict[str, Any]]:
        # Get base results from parent
        results = await super().process_new_mints(sources)
        
        # Enhance each result with RAG + AI
        for result in results:
            # RAG insights
            rag_insights = []
            if self.rag.available and result.get('symbol'):
                rag_insights = self.rag.search(f"cryptocurrency trading {result['symbol']}", k=2)
                result['rag_insights'] = rag_insights
            
            # Qwen AI analysis
            if self.qwen.available:
                result['ai_analysis'] = self.qwen.analyze(result, rag_insights)
        
        return results

# ============================================================================
# MAIN LOOP WITH CONTINUOUS MONITORING
# ============================================================================

async def run_continuous(interval: int, output_file: str, sources: List[str]):
    """Continuously monitor and scrape new tokens"""
    orch = AIOrchestrator(sources)
    
    # Initialize pump.fun monitor if enabled
    pumpfun = PumpFunMonitor() if USE_PUMPFUN_WEBSOCKET else None
    
    print("\n" + "="*80)
    print("🚀 SOLANA MEMECOIN SCRAPER - CONTINUOUS MODE")
    print("="*80)
    print(f"📁 Output: {output_file}")
    print(f"⏱️  Interval: {interval}s")
    print(f"🔍 Sources: {len(sources)} DexScreener endpoints")
    if pumpfun:
        print(f"🔥 Pump.fun: ✅ WebSocket enabled (REAL new tokens)")
    print(f"🤖 RAG: {'✅ Enabled' if orch.rag.available else '❌ Disabled'}")
    print(f"🧠 AI: {'✅ Enabled' if orch.qwen.available else '❌ Disabled'}")
    print("="*80 + "\n")
    
    iteration = 0
    total_found = 0
    
    while True:
        iteration += 1
        ts = datetime.now().isoformat()
        
        print(f"\n[{ts}] 🔄 Scan #{iteration} starting...")
        
        try:
            # Gather tokens from multiple sources in parallel
            tasks = []
            
            # Task 1: DexScreener endpoints
            tasks.append(orch.process_new_mints(sources))
            
            # Task 2: Pump.fun WebSocket (monitor for interval duration)
            if pumpfun:
                async def get_pumpfun_tokens():
                    tokens = await pumpfun.connect_and_monitor(duration_seconds=min(interval, 30))
                    # Convert to format expected by orchestrator
                    formatted = []
                    for t in tokens:
                        formatted.append({
                            'mint': t['mint'],
                            'name': t['name'],
                            'symbol': t['symbol']
                        })
                    # Process through orchestrator pipeline
                    if formatted:
                        return await orch.process_new_mints(None, formatted)
                    return []
                
                tasks.append(get_pumpfun_tokens())
            
            # Wait for all sources
            all_results = await asyncio.gather(*tasks)
            
            # Combine results from all sources
            results = []
            for res in all_results:
                if res:
                    results.extend(res)
            
            # Remove duplicates by mint address
            seen_mints = set()
            unique_results = []
            for r in results:
                mint = r.get('mint')
                if mint and mint not in seen_mints:
                    seen_mints.add(mint)
                    unique_results.append(r)
            
            results = unique_results
            
            print(f"📊 Processing {len(results)} tokens through filters...")
            
            if results:
                # Filter: VERY RELAXED criteria to get SOME results
                # 1. Volume > $500 OR Liquidity > $500 (any activity)
                # 2. Risk score < 80 (not extremely risky)
                filtered = []
                rejected_reasons = {'low_volume': 0, 'high_risk': 0, 'no_data': 0}
                
                for r in results:
                    volume = r.get('volume', 0) or 0
                    liquidity = r.get('liquidity', 0) or 0
                    risk = r.get('risk_score', 100)
                    
                    # VERY simple filter - just need SOME activity
                    has_activity = volume > 500 or liquidity > 500
                    risk_ok = risk < 80
                    
                    if has_activity and risk_ok:
                        filtered.append(r)
                    else:
                        if not has_activity:
                            rejected_reasons['low_volume'] += 1
                        if not risk_ok:
                            rejected_reasons['high_risk'] += 1
                
                print(f"✅ {len(filtered)} passed filters | ❌ Rejected: {rejected_reasons}")
                
                if filtered:
                    # Save to file (REPLACE mode - creates fresh file each run)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for r in filtered:
                            f.write(json.dumps(r) + '\n')
                    
                    total_found += len(filtered)
                    
                    # Display results
                    print(f"✅ Found {len(filtered)} tokens (Total: {total_found}):")
                    for r in filtered[:5]:  # Show first 5
                        print(f"\n{'='*80}")
                        print(f"💎 {r.get('symbol', 'UNKNOWN')} - {r.get('name', 'Unknown Token')}")
                        print(f"📍 Mint: {r['mint']}")
                        price = r.get('price') or 0.0
                        volume = r.get('volume') or 0.0
                        liquidity = r.get('liquidity') or 0.0
                        prob = r.get('profit_probability') or 0.0
                        age_days = r.get('age_days')
                        lp_locked = r.get('rug_details', {}).get('lp_locked_pct', 0)
                        dev_own = r.get('rug_details', {}).get('top_wallet_pct', 0)
                        
                        # Better age formatting
                        if age_days is not None:
                            age_seconds = age_days * 86400
                            if age_seconds < 60:
                                age_str = f"{age_seconds:.0f}s"
                            elif age_seconds < 3600:
                                age_str = f"{age_seconds/60:.1f}m"
                            else:
                                age_str = f"{age_seconds/3600:.1f}h"
                        else:
                            age_str = "Unknown"
                        
                        print(f"💰 Price: ${price:.8f}")
                        print(f"📊 Volume: ${volume:,.0f} | 🏦 Liquidity: ${liquidity:,.0f}")
                        print(f"⏰ Age: {age_str} | 🔒 LP Locked: {lp_locked:.0f}% | 👤 Dev: {dev_own:.0f}%")
                        print(f"🎯 Risk: {r.get('risk_score', 50)}/100 | Profit: {prob*100:.2f}%")
                    
                    if len(filtered) > 5:
                        print(f"\n... and {len(filtered) - 5} more tokens (check {output_file})")
                else:
                    print(f"⚠️  Found {len(results)} tokens but none passed filters")
            else:
                print("⚠️  No new tokens discovered")
        
        except asyncio.CancelledError:
            print("\n🛑 Scraper stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in scan #{iteration}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n⏳ Next scan in {interval}s... (Total so far: {total_found} tokens)\n")
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            print("\n🛑 Scraper stopped by user")
            break

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='🚀 Solana Memecoin Scraper with AI')
    parser.add_argument('--interval', type=int, default=60, help='Scan interval in seconds (default: 60)')
    parser.add_argument('--out', type=str, default='scraped_tokens.json', help='Output file')
    parser.add_argument('--sources', type=str, nargs='*', help='Custom source endpoints')
    args = parser.parse_args()

    sources = args.sources if args.sources else DEFAULT_SOURCES
    
    if not HAS_AI:
        print("⚠️  RAG/AI features disabled. Install: pip install faiss-cpu sentence-transformers langchain-community langchain-huggingface ollama\n")
    
    asyncio.run(run_continuous(args.interval, args.out, sources))

if __name__ == '__main__':
    main()
