import os
import random
import requests
from bitcoin import SelectParams
from bitcoin.wallet import CBitcoinSecret, P2PKHBitcoinAddress
from bitcoin.core import b2lx, lx
from bitcoin.core.script import CScript
from bitcoin.core.scripteval import VerifyScript, SCRIPT_VERIFY_P2SH
from bitcoin.rpc import Proxy


def choose_network(context):
    if context == 'real_transaction':
        return 'mainnet'
    elif context == 'testing':
        return 'testnet'
    else:
        raise ValueError("Invalid context. Choose 'real_transaction' or 'testing'.")


def create_wallet(network='mainnet'):
    SelectParams(network)
    secret = CBitcoinSecret.from_secret_bytes(os.urandom(32))
    address = P2PKHBitcoinAddress.from_pubkey(secret.pub)
    return str(address), str(secret)


def get_balance(address, network='mainnet'):
    SelectParams(network)
    if network == 'mainnet':
        url = f'https://blockchain.info/q/addressbalance/{address}?confirmations=6'
    elif network == 'testnet':
        url = f'https://api.blockcypher.com/v1/btc/test3/addrs/{address}/balance'

    response = requests.get(url)
    if response.status_code == 200:
        if network == 'mainnet':
            return int(response.text) / 100000000  # Convert from satoshis to BTC
        elif network == 'testnet':
            return response.json()['balance'] / 100000000
    else:
        raise Exception(f'Error fetching balance: {response.text}')


def get_address(secret):
    secret_key = CBitcoinSecret(secret)
    address = P2PKHBitcoinAddress.from_pubkey(secret_key.pub)
    return str(address)


def send_bitcoin(sender_secret, recipient_address, amount_btc, fee_btc, context):
    network = choose_network(context)
    SelectParams(network)
    rpc = Proxy()

    # Rest of the send_bitcoin function implementation
