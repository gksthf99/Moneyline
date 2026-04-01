#!/usr/bin/env python3
"""
Batch-redeem all settled winning positions on Polymarket.

Collects all redeemable conditions, packs them into a single
Gnosis Safe MultiSend transaction, and executes once on-chain.

One tx redeems everything. Runs before the daily executor.

Usage:
    python scripts/redeem_positions.py              # redeem all
    python scripts/redeem_positions.py --dry-run     # show what would be redeemed
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from web3 import Web3, Account
from eth_account.messages import defunct_hash_message
from eth_abi import encode as abi_encode
from py_clob_client.config import get_contract_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
POLYGON_RPC = os.getenv("POLYGON_RPC", "https://polygon-bor-rpc.publicnode.com")

# Gnosis Safe MultiSend on Polygon
MULTISEND_ADDR = "0xA238CBeb142c10Ef7Ad8442C6D1f9E89e07e7761"

# Safe tx typehash for EIP-712
SAFE_TX_TYPEHASH = bytes.fromhex(
    "bb8310d486368db6bd6f849402fdd73ad53d316b5a4b2644ad6efe0f941286d8"
)

SAFE_EXEC_ABI = [
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "signatures", "type": "bytes"},
        ],
        "name": "execTransaction",
        "outputs": [{"name": "success", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "nonce",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "domainSeparator",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
]

CTF_REDEEM_SELECTOR = bytes.fromhex("01b7037c")  # redeemPositions(address,bytes32,bytes32,uint256[])

BALANCE_OF_ABI = [
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_winning_trades() -> list[dict]:
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/trades",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        params={
            "select": "id,game_id,team_picked,token_id,amount_usdc,entry_price,outcome",
            "outcome": "eq.win",
            "token_id": "not.is.null",
            "order": "created_at",
        },
        timeout=10,
    )
    return resp.json() if resp.status_code == 200 else []


def get_condition_id(token_id: str) -> str:
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"clob_token_ids": token_id},
            timeout=10,
        )
        if resp.status_code == 200 and resp.json():
            return resp.json()[0].get("conditionId", "")
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Calldata builders
# ---------------------------------------------------------------------------

def build_redeem_calldata(collateral: str, condition_id: str) -> bytes:
    """Encode CTF.redeemPositions(collateral, 0x0, conditionId, [1,2])."""
    args = abi_encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [
            Web3.to_checksum_address(collateral),
            b"\x00" * 32,
            bytes.fromhex(condition_id.replace("0x", "")),
            [1, 2],
        ],
    )
    return CTF_REDEEM_SELECTOR + args


def build_multisend_data(calls: list[tuple[str, bytes]]) -> bytes:
    """Pack multiple (to, calldata) pairs into MultiSend.multiSend(bytes) payload.

    Each sub-tx is packed as: operation(1) + to(20) + value(32) + dataLength(32) + data
    """
    packed = b""
    for to_addr, data in calls:
        addr_bytes = bytes.fromhex(to_addr.replace("0x", ""))
        packed += (
            b"\x00"                                     # operation = CALL
            + addr_bytes                                # to (20 bytes)
            + (0).to_bytes(32, "big")                   # value = 0
            + len(data).to_bytes(32, "big")             # data length
            + data                                      # data
        )
    # multiSend(bytes transactions) selector = 0x8d80ff0a
    return bytes.fromhex("8d80ff0a") + abi_encode(["bytes"], [packed])


# ---------------------------------------------------------------------------
# Safe signing & execution
# ---------------------------------------------------------------------------

def sign_and_exec(
    w3: Web3,
    safe_addr: str,
    to: str,
    data: bytes,
    operation: int,
    private_key: str,
) -> str | None:
    """Sign a Safe tx with the EOA and submit on-chain.

    Uses the Gnosis Safe EIP-712 signing scheme with the exact
    safeTxHash computation matching the Safe contract.
    """
    safe = w3.eth.contract(
        address=Web3.to_checksum_address(safe_addr), abi=SAFE_EXEC_ABI
    )
    safe_nonce = safe.functions.nonce().call()
    domain_sep = safe.functions.domainSeparator().call()

    zero_addr = "0x" + "00" * 20
    data_hash = Web3.keccak(primitive=data)

    # Safe's encodeTransactionData — must match contract exactly
    # The contract does: keccak256(abi.encode(SAFE_TX_TYPEHASH, to, value, keccak256(data),
    #   operation, safeTxGas, baseGas, gasPrice, gasToken, refundReceiver, _nonce))
    encoded = abi_encode(
        [
            "bytes32", "address", "uint256", "bytes32", "uint8",
            "uint256", "uint256", "uint256", "address", "address", "uint256",
        ],
        [
            SAFE_TX_TYPEHASH,
            Web3.to_checksum_address(to),
            0,              # value
            data_hash,
            operation,
            0,              # safeTxGas
            0,              # baseGas
            0,              # gasPrice
            zero_addr,      # gasToken
            zero_addr,      # refundReceiver
            safe_nonce,
        ],
    )
    struct_hash = Web3.keccak(primitive=encoded)

    # EIP-712: \x19\x01 + domainSeparator + structHash
    safe_tx_hash = Web3.keccak(primitive=b"\x19\x01" + domain_sep + struct_hash)

    # Sign the hash
    acct = Account.from_key(private_key)
    sig_obj = acct.unsafe_sign_hash(safe_tx_hash)

    # Gnosis Safe expects v = 27 or 28 (not 0/1)
    v = sig_obj.v
    if v < 27:
        v += 27

    signature = sig_obj.r.to_bytes(32, "big") + sig_obj.s.to_bytes(32, "big") + v.to_bytes(1, "big")

    # Build and send the on-chain tx
    eoa = acct.address
    tx = safe.functions.execTransaction(
        Web3.to_checksum_address(to),
        0,              # value
        data,
        operation,
        0, 0, 0,        # safeTxGas, baseGas, gasPrice
        zero_addr,       # gasToken
        zero_addr,       # refundReceiver
        signature,
    ).build_transaction({
        "from": eoa,
        "nonce": w3.eth.get_transaction_count(eoa),
        "gas": 800_000,
        "gasPrice": w3.eth.gas_price,
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    logger.info("  Tx sent: %s — waiting...", tx_hash.hex())

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status == 1:
        logger.info("  Confirmed in block %d (gas: %d)", receipt.blockNumber, receipt.gasUsed)
        return tx_hash.hex()
    else:
        logger.error("  REVERTED: %s (GS013=bad sig, GS026=bad module)", tx_hash.hex())
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False):
    private_key = os.getenv("POLY_PRIVATE_KEY", "")
    proxy = os.getenv("POLY_FUNDER_ADDRESS", "")
    if not private_key or not proxy:
        logger.error("POLY_PRIVATE_KEY and POLY_FUNDER_ADDRESS required")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    if not w3.is_connected():
        logger.error("Cannot connect to Polygon RPC")
        sys.exit(1)

    config = get_contract_config(137)
    ctf_address = config.conditional_tokens
    collateral = config.collateral

    ctf = w3.eth.contract(address=Web3.to_checksum_address(ctf_address), abi=BALANCE_OF_ABI)

    # Collect redeemable positions
    winning = get_winning_trades()
    if not winning:
        logger.info("No winning trades to redeem")
        return

    redeem_calls: list[tuple[str, bytes]] = []
    seen_conditions: set[str] = set()
    total_value = 0.0

    for trade in winning:
        token_id = trade["token_id"]
        balance = ctf.functions.balanceOf(
            Web3.to_checksum_address(proxy), int(token_id)
        ).call()
        if balance == 0:
            continue

        shares = balance / 1e6
        total_value += shares

        condition_id = get_condition_id(token_id)
        if not condition_id or condition_id in seen_conditions:
            if not condition_id:
                logger.warning("Trade %d (%s): no condition_id — skipping", trade["id"], trade["team_picked"])
            continue
        seen_conditions.add(condition_id)

        logger.info(
            "Trade %d: %s | %.2f shares ($%.2f) | condition %s...",
            trade["id"], trade["team_picked"], shares, shares, condition_id[:16],
        )
        redeem_calls.append((ctf_address, build_redeem_calldata(collateral, condition_id)))

    if not redeem_calls:
        logger.info("Nothing to redeem (all already claimed)")
        return

    logger.info("Batching %d redemptions into 1 transaction (~$%.2f total)", len(redeem_calls), total_value)

    if dry_run:
        logger.info("DRY RUN — no transaction sent")
        return

    # Single call if only 1 redemption, MultiSend if >1
    if len(redeem_calls) == 1:
        to, data = redeem_calls[0]
        tx_hash = sign_and_exec(w3, proxy, to, data, 0, private_key)  # operation=0 (CALL)
    else:
        ms_data = build_multisend_data(redeem_calls)
        tx_hash = sign_and_exec(w3, proxy, MULTISEND_ADDR, ms_data, 1, private_key)  # operation=1 (DELEGATECALL)

    if tx_hash:
        logger.info("ALL REDEEMED: %d conditions in tx %s", len(redeem_calls), tx_hash)
    else:
        logger.error("Batch redemption failed")


if __name__ == "__main__":
    run(dry_run="--dry-run" in sys.argv)
