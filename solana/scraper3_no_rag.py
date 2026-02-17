"""
🚀 SOLANA MEMECOIN SCRAPER - Simplified (No RAG)
Focus: Get FULL market data from DexScreener proper API

Usage:
  python scraper3_no_rag.py --interval 10
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# Import from local modules
from orchestrator import Orchestrator

# ============================================================================
# TOKEN DISCOVERY SOURCES
# ============================================================================

DEFAULT_SOURCES = [
    "https://api.dexscreener.com/token-profiles/latest/v1",
    "https://api.dexscreener.com/token-boosts/latest/v1",
]

# ============================================================================
# SIMPLIFIED ORCHESTRATOR (No RAG/AI)
# ============================================================================

class SimpleOrchestrator(Orchestrator):
    """Orchestrator without RAG/AI - just pure market data"""
    pass

# ============================================================================
# CONTINUOUS SCRAPING LOGIC
# ============================================================================

async def run_continuous(interval: int = 60):
    """Main continuous loop"""
    orch = SimpleOrchestrator(sources=DEFAULT_SOURCES)
    output_path = os.path.join(os.path.dirname(__file__), 'scraped_tokens.json')
    scan_count = 0
    
    print(f"\n{'='*80}")
    print(f"🚀 SOLANA SCRAPER STARTED (NO RAG)")
    print(f"✅ Using DexScreener proper API for FULL market data")
    print(f"📊 Scan interval: {interval}s")
    print(f"📁 Output: {output_path}")
    print('='*80)
    
    while True:
        scan_count += 1
        print(f"\n\n🔄 SCAN #{scan_count} [{datetime.utcnow().isoformat()}Z]")
        print("-" * 80)
        
        try:
            # Process new tokens
            results = await orch.process_new_mints()
            
            if not results:
                print("⚠️  No new tokens discovered")
                await asyncio.sleep(interval)
                continue
            
            # Filter: volume > $100 OR liquidity > $100 OR risk < 80%
            filtered = []
            for r in results:
                vol = r.get('volume', 0) or 0
                liq = r.get('liquidity', 0) or 0
                risk = r.get('risk_score', 100)
                if vol > 100 or liq > 100 or risk < 80:
                    filtered.append(r)
            
            if not filtered:
                print(f"✅ Found {len(results)} tokens (Total: {len(results)})")
                print("⚠️  All tokens filtered out (volume/liquidity too low)")
                await asyncio.sleep(interval)
                continue
            
            print(f"✅ Found {len(filtered)} tokens (Total: {len(filtered)})")
            
            # Save to file (JSON-lines format)
            with open(output_path, 'a', encoding='utf-8') as f:
                for token in filtered:
                    f.write(json.dumps(token) + '\n')
            
            # Display first 5 tokens
            print(f"\n📊 TOP {min(5, len(filtered))} TOKENS:")
            for i, r in enumerate(filtered[:5], 1):
                price = r.get('price') or 0.0
                volume = r.get('volume') or 0.0
                liquidity = r.get('liquidity') or 0.0
                prob = r.get('profit_probability') or 0.0
                
                print(f"\n{i}. {r.get('symbol', 'UNKNOWN')} - {r.get('name', 'Unknown')}")
                print(f"   Mint: {r['mint']}")
                print(f"   Price: ${price:.8f} | Vol: ${volume:,.0f} | Liq: ${liquidity:,.0f}")
                print(f"   Risk: {r.get('risk_score', 0):.0f}% | Probability: {prob*100:.1f}% | Score: {r.get('final_score', 0):.3f}")
            
            print(f"\n💾 Saved {len(filtered)} tokens to {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"❌ Error in scan: {str(e)}")
        
        # Wait for next scan
        print(f"\n⏳ Next scan in {interval}s...")
        await asyncio.sleep(interval)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Solana token scraper (No RAG)")
    parser.add_argument('--interval', type=int, default=60, help='Scan interval in seconds')
    args = parser.parse_args()
    
    try:
        asyncio.run(run_continuous(interval=args.interval))
    except KeyboardInterrupt:
        print("\n\n🛑 Scraper stopped by user")

if __name__ == '__main__':
    main()
