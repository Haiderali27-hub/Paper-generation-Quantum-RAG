"""Test DexScreener proper API to verify we get full market data"""
import asyncio
import aiohttp

async def test_dexscreener_api():
    # Test with one of the tokens from your scraped_tokens.json
    test_mint = "BA7uo4a7qRFa2W1uUNPY3EL6Fr8w1ysBut9ivjpwpump"
    
    url = f"https://api.dexscreener.com/latest/dex/tokens/{test_mint}"
    
    print(f"\n{'='*80}")
    print(f"Testing DexScreener Proper API")
    print(f"Mint: {test_mint}")
    print(f"URL: {url}")
    print('='*80)
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            print(f"\nStatus: {resp.status}")
            
            if resp.status == 200:
                data = await resp.json()
                pairs = data.get('pairs', [])
                
                print(f"\nFound {len(pairs)} trading pairs")
                
                if pairs:
                    # Show best pair (highest liquidity)
                    best_pair = max(pairs, key=lambda p: float(p.get('liquidity', {}).get('usd', 0) or 0))
                    
                    print(f"\n✅ BEST PAIR (Highest Liquidity):")
                    print(f"  DEX: {best_pair.get('dexId', 'Unknown')}")
                    print(f"  Pair Address: {best_pair.get('pairAddress', 'Unknown')}")
                    
                    base_token = best_pair.get('baseToken', {})
                    print(f"\n  Token Info:")
                    print(f"    Name: {base_token.get('name', 'N/A')}")
                    print(f"    Symbol: {base_token.get('symbol', 'N/A')}")
                    
                    print(f"\n  Market Data:")
                    print(f"    Price: ${float(best_pair.get('priceUsd', 0) or 0):.8f}")
                    print(f"    Volume 24h: ${float(best_pair.get('volume', {}).get('h24', 0) or 0):,.2f}")
                    print(f"    Liquidity: ${float(best_pair.get('liquidity', {}).get('usd', 0) or 0):,.2f}")
                    print(f"    Price Change 24h: {best_pair.get('priceChange', {}).get('h24', 0)}%")
                    
                    print(f"\n  Trading Activity:")
                    print(f"    Txns 24h: {best_pair.get('txns', {}).get('h24', {}).get('buys', 0) + best_pair.get('txns', {}).get('h24', {}).get('sells', 0)}")
                    print(f"    Buys: {best_pair.get('txns', {}).get('h24', {}).get('buys', 0)}")
                    print(f"    Sells: {best_pair.get('txns', {}).get('h24', {}).get('sells', 0)}")
                    
                    print(f"\n\u2705 SUCCESS: DexScreener API provides FULL market data!")
                else:
                    print(f"\n\u26a0\ufe0f  No trading pairs found for this token")
            else:
                text = await resp.text()
                print(f"\n\u274c Error: {text[:300]}")

if __name__ == '__main__':
    asyncio.run(test_dexscreener_api())
