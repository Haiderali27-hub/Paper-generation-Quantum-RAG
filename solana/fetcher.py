"""
Fetcher module: discovers new mints and prevents duplicates
Uses async HTTP requests and persists seen mints to JSON
"""
import asyncio
import aiohttp
import json
import os
from typing import List, Dict, Any

SEEN_FILE = os.path.join(os.path.dirname(__file__), 'seen_mints.json')

class Fetcher:
    def __init__(self, sources: List[str]=None):
        self.sources = sources or []
        self.seen = set()
        self._load_seen()

    def _load_seen(self):
        if os.path.exists(SEEN_FILE):
            try:
                with open(SEEN_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.seen = set(data.get('seen', []))
            except Exception:
                self.seen = set()
        else:
            self.seen = set()

    def _save_seen(self):
        try:
            with open(SEEN_FILE, 'w', encoding='utf-8') as f:
                json.dump({'seen': list(self.seen)}, f)
        except Exception:
            pass

    async def fetch_from_source(self, session: aiohttp.ClientSession, source: str) -> List[Dict[str, Any]]:
        """Fetch tokens from DexScreener v1 API and extract Solana addresses"""
        try:
            print(f"🌐 Fetching: {source[:60]}...")
            async with session.get(source, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"❌ Error {resp.status}: {text[:200]}")
                    return []
                data = await resp.json()
                
                results = []
                
                # DexScreener v1 API - returns array of token objects
                if isinstance(data, list):
                    print(f"📊 Found {len(data)} items")
                    for item in data:
                        if isinstance(item, dict):
                            # Extract chainId and tokenAddress
                            chain_id = item.get('chainId', '')
                            token_addr = item.get('tokenAddress', '')
                            
                            # Only process Solana tokens
                            if chain_id == 'solana' and token_addr:
                                results.append({
                                    'mint': token_addr,
                                    'name': item.get('description', 'Unknown'),
                                    'symbol': 'UNKNOWN',
                                    'url': item.get('url', ''),
                                })
                
                print(f"✅ Extracted {len(results)} Solana tokens")
                return results
                
        except asyncio.CancelledError:
            raise  # Re-raise to properly handle cancellation
        except Exception as e:
            print(f"❌ Exception: {str(e)[:150]}")
            return []

    async def discover_mints(self, batch_sources: List[str]=None) -> List[Dict[str, Any]]:
        """Discover mints from provided sources and return unique new mints."""
        sources = batch_sources or self.sources
        new_mints = []
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_from_source(session, s) for s in sources]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, Exception):
                        continue
                    for item in res:
                        mint = item.get('mint')
                        if not mint:
                            continue
                        # DISABLED: Skip duplicate filtering - reprocess all tokens
                        # This allows tokens that previously failed filters to be re-evaluated
                        # if mint in self.seen:
                        #     continue
                        # self.seen.add(mint)
                        new_mints.append(item)
        except asyncio.CancelledError:
            pass  # Gracefully handle cancellation
        except Exception as e:
            print(f"❌ Discovery error: {e}")
        # persist seen mints
        self._save_seen()
        return new_mints
