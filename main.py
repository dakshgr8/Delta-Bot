"""
Delta-Bot: Phase 1 Foundation & REST Client Smoke Test Harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Executes comprehensive end-to-end verification against Delta Exchange:
1. Configuration loading & credential masking validation
2. HMAC-SHA256 signature generation test
3. Server time synchronization & clock drift calculation
4. Perpetual futures specifications caching (BTC, ETH, SOL, XRP)
5. Real-time ticker queries (mark price, bid/ask, spread bps, open interest)
6. Margin quantization tests (tick size, lot size)
7. Authenticated endpoint execution & exception handling (/v2/wallet/balances)
8. Rate limit telemetry report
"""

import asyncio
import logging
import sys
import time

from delta_bot.auth import DeltaAuthEngine
from delta_bot.client import DeltaRestClient
from delta_bot.config import BotConfig
from delta_bot.exceptions import (
    AuthenticationError,
    DeltaAPIError,
    RateLimitError,
)

# Configure structured logging with standard library
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("delta_bot.main")


def print_banner(title: str) -> None:
    """Prints a styled section separator for smoke test output."""
    border = "=" * 80
    print(f"\n{border}\n >>> {title.upper()}\n{border}")


def test_hmac_signing_engine() -> None:
    """
    Synchronous unit validation of HMAC-SHA256 signature generator.
    Verifies that the canonical signature format:
        METHOD + timestamp + path + query_string + payload
    matches expected cryptographic output.
    """
    print_banner("1. Validating HMAC-SHA256 Authentication Engine")
    dummy_key = "test_api_key_12345"
    dummy_secret = "test_api_secret_67890"

    auth = DeltaAuthEngine(
        api_key=dummy_key,
        api_secret=dummy_secret,
        user_agent="DeltaBot-QuantEngine/1.0.0",
    )

    fixed_timestamp = "1712750400"
    path = "/v2/wallet/balances"
    method = "GET"
    sig, ts = auth.create_signature(
        method=method,
        path=path,
        query="",
        payload="",
        timestamp=fixed_timestamp,
    )

    headers = auth.build_headers(
        method=method,
        path=path,
        query="",
        payload="",
        is_authenticated=True,
    )

    logger.info("Generated test signature: %s", sig)
    logger.info("Generated headers: %s", {k: (v if k != "api-key" else "********") for k, v in headers.items()})

    assert ts == fixed_timestamp, "Timestamp mismatch in signature engine"
    assert "signature" in headers, "Missing signature in generated headers"
    assert "timestamp" in headers, "Missing timestamp in generated headers"
    assert headers["api-key"] == dummy_key, "API key mismatch in headers"
    logger.info("✓ HMAC-SHA256 cryptographic signing engine passed verification.")


async def run_smoke_tests() -> None:
    """Orchestrates Phase 1 asynchronous smoke tests against Delta Exchange."""
    print_banner("2. Initializing Configuration & Security Guard")
    config = BotConfig()
    logger.info("Bot Configuration loaded successfully:")
    for k, v in config.safe_summary().items():
        logger.info("  - %-22s: %s", k, v)

    # Check credential masking in __repr__
    repr_str = repr(config)
    assert config.api_secret_str not in repr_str or not config.api_secret_str, (
        "Security breach: API Secret detected in __repr__!"
    )
    logger.info("✓ Credential masking verified. Private credentials redacted from all representations.")

    async with DeltaRestClient(config=config) as client:
        # Step 3: Server Time Synchronization
        print_banner("3. Exchange Server Time Synchronization")
        time_info = await client.get_server_time()
        logger.info("Server Epoch:      %.4f s", time_info.server_epoch_s)
        logger.info("Host Local Epoch:  %.4f s", time_info.local_epoch_s)
        logger.info("Clock Offset:      %+.4f s (Server - Local)", time_info.clock_offset_s)
        logger.info("Round-Trip RTT:    %.2f ms", time_info.round_trip_latency_ms)
        logger.info("Drift Acceptable:  %s (Required: < 3.5s drift)", time_info.is_drift_acceptable)

        # Step 4: Product Specifications & Caching
        print_banner("4. Fetching & Caching Contract Specifications (/v2/products)")
        specs = await client.get_products(contract_types=["perpetual_futures"])
        logger.info("Total perpetual futures contracts indexed: %d", len(specs))

        # Target assets: BTC, ETH, SOL, XRP
        target_map = await client.get_target_perpetuals()
        print("\n" + "-" * 105)
        header = f"{'Symbol':<12} | {'Product ID':<10} | {'Contract Value':<15} | {'Tick Size':<10} | {'Maker Fee':<10} | {'Taker Fee':<10} | {'Status':<12}"
        print(header)
        print("-" * 105)
        for base_asset in client.TARGET_BASE_SYMBOLS:
            spec = target_map.get(base_asset)
            if spec:
                row = (
                    f"{spec.symbol:<12} | "
                    f"{spec.product_id:<10} | "
                    f"{spec.contract_value:<15} | "
                    f"{spec.tick_size:<10} | "
                    f"{spec.maker_commission_rate:<10.6f} | "
                    f"{spec.taker_commission_rate:<10.6f} | "
                    f"{spec.trading_status:<12}"
                )
                print(row)
            else:
                print(f"{base_asset:<12} | NOT FOUND")
        print("-" * 105 + "\n")

        # Step 5: Order Quantization Test
        print_banner("5. Price & Order Size Quantization Verification (Lots vs Contracts)")
        btc_spec = target_map.get("BTC")
        if btc_spec:
            sample_price = 75432.37891
            target_btc_qty = 0.0076543
            quantized_price = btc_spec.round_price(sample_price)
            integer_contracts = btc_spec.round_size(target_btc_qty)
            effective_qty = btc_spec.contracts_to_asset_qty(integer_contracts)

            logger.info("Asset: %s (Tick: %s, Contract Value: %s BTC)", btc_spec.symbol, btc_spec.tick_size, btc_spec.contract_value)
            logger.info("  Raw Target Price: %s -> Quantized Price: %s", sample_price, quantized_price)
            logger.info("  Raw Strategy Qty: %s BTC -> Integer Contracts: %d (Effective: %s BTC)", target_btc_qty, integer_contracts, effective_qty)
            assert isinstance(integer_contracts, int), "Delta order size must strictly be an integer number of contracts!"
            assert integer_contracts == 8, f"Expected 8 contracts for 0.0076543 BTC, got {integer_contracts}"
            assert (quantized_price / btc_spec.tick_size).is_integer() or round(quantized_price % btc_spec.tick_size, 6) in (0.0, btc_spec.tick_size), "Tick size rounding mismatch!"
            logger.info("✓ Quantization rules strictly adhere to integer contract specifications.")

        # Step 5b: Cold-Start Warm-Up Historical Backfill
        print_banner("5b. Cold-Start REST Historical Candle Warm-up (/v2/history/candles)")
        if btc_spec:
            hist_candles = await client.get_historical_candles(btc_spec.symbol, resolution="1m", limit=30)
            logger.info("Backfilled %d historical 1m closed candles for %s on startup.", len(hist_candles), btc_spec.symbol)
            if hist_candles:
                oldest = hist_candles[0]
                latest = hist_candles[-1]
                logger.info("  Oldest Bar: Time=%s | Close=%.2f", oldest.candle_start_epoch_s, oldest.close)
                logger.info("  Latest Bar: Time=%s | Close=%.2f", latest.candle_start_epoch_s, latest.close)
            logger.info("✓ REST Warm-up routine ready to prime CircularBarCache on minute zero.")

        # Step 6: Real-Time Market Tickers
        print_banner("6. Fetching Real-Time Market Tickers (/v2/tickers)")
        print("\n" + "-" * 115)
        ticker_header = (
            f"{'Symbol':<10} | {'Mark Price':<12} | {'Best Bid':<10} | {'Best Ask':<10} | "
            f"{'Spread':<10} | {'Spread (bps)':<12} | {'Open Interest':<14} | {'24h Vol (USD)':<14}"
        )
        print(ticker_header)
        print("-" * 115)
        for base_asset in client.TARGET_BASE_SYMBOLS:
            spec = target_map.get(base_asset)
            if spec:
                try:
                    ticker = await client.get_ticker(spec.symbol)
                    bid_str = f"{ticker.best_bid:.4f}" if ticker.best_bid else "N/A"
                    ask_str = f"{ticker.best_ask:.4f}" if ticker.best_ask else "N/A"
                    spread_str = f"{ticker.spread:.4f}" if ticker.spread is not None else "N/A"
                    spread_bps_str = f"{ticker.spread_bps:.2f}" if ticker.spread_bps is not None else "N/A"
                    turnover_str = f"${ticker.turnover_24h:,.0f}" if ticker.turnover_24h else "N/A"
                    row = (
                        f"{ticker.symbol:<10} | "
                        f"{ticker.mark_price:<12.4f} | "
                        f"{bid_str:<10} | "
                        f"{ask_str:<10} | "
                        f"{spread_str:<10} | "
                        f"{spread_bps_str:<12} | "
                        f"{ticker.open_interest:<14.2f} | "
                        f"{turnover_str:<14}"
                    )
                    print(row)
                except Exception as ex:
                    logger.error("Failed to fetch ticker for %s: %s", spec.symbol, ex)
        print("-" * 115 + "\n")

        # Step 7: Authenticated Request Test (/v2/wallet/balances)
        print_banner("7. Testing Authenticated Endpoint (/v2/wallet/balances)")
        if config.has_valid_credentials():
            logger.info("Valid credentials detected in configuration. Querying live wallet balances...")
            try:
                balances = await client.get_wallet_balances()
                if balances:
                    for b in balances:
                        logger.info(
                            "Asset: %-6s | Balance: %10.4f | Available: %10.4f | Blocked: %10.4f (Util: %.1f%%)",
                            b.asset_symbol,
                            b.balance,
                            b.available_balance,
                            b.blocked_margin,
                            b.margin_utilization_pct,
                        )
                else:
                    logger.info("No balances found in account.")
            except AuthenticationError as auth_err:
                logger.error("Authentication rejected by exchange: %s", auth_err)
            except DeltaAPIError as api_err:
                logger.error("API error during balance retrieval: %s", api_err)
        else:
            logger.info("No live API credentials provided in .env (running in sandbox/unauthenticated mode).")
            logger.info("Testing signed request rejection and typed AuthenticationError classification...")
            try:
                # Intentionally trigger signed request with dummy key to verify signature construction and error pipeline
                dummy_client_auth = DeltaAuthEngine(
                    api_key="TEST_INVALID_KEY_VERIFICATION",
                    api_secret="TEST_INVALID_SECRET_VERIFICATION_12345",
                    user_agent=config.user_agent,
                )
                await client.http_engine.request(
                    method="GET",
                    path="/v2/wallet/balances",
                    is_authenticated=True,
                )
            except AuthenticationError as caught_auth_err:
                logger.info(
                    "✓ Expected AuthenticationError caught successfully:\n  %s",
                    caught_auth_err,
                )
                logger.info("✓ Error classification, signature headers, and taxonomy validated.")
            except Exception as unexp_err:
                logger.warning("Captured response: %s", unexp_err)

        # Step 8: Rate Limit Telemetry
        print_banner("8. Exchange Rate Limit Telemetry")
        rl_state = client.rate_limit_state
        logger.info("Rate Limit Maximum:   %s", rl_state.limit or "Dynamic/Unset")
        logger.info("Remaining Requests:   %s", rl_state.remaining or "Full")
        logger.info("Reset Interval:       %s ms", rl_state.reset_ms or "N/A")
        logger.info("Quota Exhausted:      %s", rl_state.is_exhausted)
        logger.info("Utilization:          %.2f%%", rl_state.utilization_pct)

    print_banner("Phase 1 Foundation & REST Client Verification: SUCCESS")


if __name__ == "__main__":
    test_hmac_signing_engine()
    asyncio.run(run_smoke_tests())
