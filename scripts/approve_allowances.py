"""
One-time token allowance approval for Polymarket trading.

Approves USDC and Conditional Tokens for the three Polymarket contracts
on Polygon. Must be run once before placing any orders.

Usage: python3 scripts/approve_allowances.py
"""

import os
import sys
import json
import time

from dotenv import load_dotenv
load_dotenv()

from eth_account import Account
from web3 import Web3

# Polygon RPC
RPC_URL = "https://polygon.drpc.org"

# Polymarket contracts that need approval
EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# USDC on Polygon
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# Conditional Tokens (CTF)
CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Max approval
MAX_UINT256 = 2**256 - 1

# ERC20 approve ABI
ERC20_ABI = json.loads('[{"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"stateMutability":"nonpayable","type":"function"}]')

# ERC1155 setApprovalForAll ABI
ERC1155_ABI = json.loads('[{"inputs":[{"name":"operator","type":"address"},{"name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"}]')


def main():
    private_key = os.getenv("POLY_PRIVATE_KEY", "")
    if not private_key:
        print("ERROR: POLY_PRIVATE_KEY not set in .env")
        sys.exit(1)

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print("ERROR: Cannot connect to Polygon RPC")
        sys.exit(1)

    account = Account.from_key(private_key)
    address = account.address
    print(f"Wallet: {address}")

    balance = w3.eth.get_balance(address)
    matic = w3.from_wei(balance, "ether")
    print(f"MATIC balance: {matic:.4f} (needed for gas)")

    if balance == 0:
        print("WARNING: No MATIC for gas fees. Send ~0.1 MATIC to your wallet first.")
        sys.exit(1)

    spenders = [
        ("Exchange", EXCHANGE),
        ("Neg Risk Exchange", NEG_RISK_EXCHANGE),
        ("Neg Risk Adapter", NEG_RISK_ADAPTER),
    ]

    # Approve USDC for all three contracts
    usdc_contract = w3.eth.contract(address=Web3.to_checksum_address(USDC), abi=ERC20_ABI)
    for name, spender in spenders:
        print(f"\nApproving USDC for {name}...")
        try:
            tx = usdc_contract.functions.approve(
                Web3.to_checksum_address(spender),
                MAX_UINT256,
            ).build_transaction({
                "from": address,
                "nonce": w3.eth.get_transaction_count(address),
                "gas": 60000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  TX: {tx_hash.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            print(f"  Status: {'OK' if receipt['status'] == 1 else 'FAILED'}")
            time.sleep(2)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Approve Conditional Tokens (ERC1155) for all three contracts
    ctf_contract = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=ERC1155_ABI)
    for name, spender in spenders:
        print(f"\nApproving CTF for {name}...")
        try:
            tx = ctf_contract.functions.setApprovalForAll(
                Web3.to_checksum_address(spender),
                True,
            ).build_transaction({
                "from": address,
                "nonce": w3.eth.get_transaction_count(address),
                "gas": 60000,
                "gasPrice": w3.eth.gas_price,
                "chainId": 137,
            })
            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  TX: {tx_hash.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            print(f"  Status: {'OK' if receipt['status'] == 1 else 'FAILED'}")
            time.sleep(2)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nAll approvals complete. You can now trade on Polymarket.")


if __name__ == "__main__":
    main()
