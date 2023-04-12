import os
import requests
from bitcoin import SelectParams
from bitcoin.wallet import CBitcoinSecret, P2PKHBitcoinAddress
from bitcoin.core import b2lx, lx
from bitcoin.core.script import CScript
from bitcoin.core.scripteval import VerifyScript, SCRIPT_VERIFY_P2SH
from bitcoin.rpc import Proxy


def create_wallet():
    secret = CBitcoinSecret.from_secret_bytes(os.urandom(32))
    address = P2PKHBitcoinAddress.from_pubkey(secret.pub)
    return str(address), str(secret)


def get_balance(address):
    url = f'https://blockchain.info/q/addressbalance/{address}?confirmations=6'
    response = requests.get(url)
    if response.status_code == 200:
        return int(response.text) / 100000000  # Convert from satoshis to BTC
    else:
        raise Exception(f'Error fetching balance: {response.text}')


def send_bitcoin(sender_secret, recipient_address, amount_btc, fee_btc):
    SelectParams('mainnet')
    rpc = Proxy()

    sender_secret = CBitcoinSecret(sender_secret)
    sender_address = P2PKHBitcoinAddress.from_pubkey(sender_secret.pub)
    recipient_address = P2PKHBitcoinAddress(recipient_address)

    amount_satoshis = int(amount_btc * 100000000)
    fee_satoshis = int(fee_btc * 100000000)

    unspent = [utxo for utxo in rpc.listunspent(0) if utxo['address'] == sender_address]
    if not unspent:
        raise Exception('No available UTXOs for the sender address.')

    unspent.sort(key=lambda x: -x['amount'])
    #randomise UTXOs
    random.shuffle(unspent)
    tx_ins = []
    tx_outs = []
    total_input_value = 0
    for utxo in unspent:
        txid = lx(utxo['txid'].hex())
        vout = utxo['vout']
        script_pub_key = utxo['scriptPubKey']
        input_value = utxo['amount']

        total_input_value += input_value

        tx_ins.append(rpc.create_raw_input(txid, vout))
        tx_outs.append((script_pub_key, input_value))

        if total_input_value >= (amount_satoshis + fee_satoshis):
            change_value = total_input_value - (amount_satoshis + fee_satoshis)
            break
    else:
        raise Exception('Insufficient funds.')

    if change_value > 0:
        tx_outs.append((CScript([sender_secret.pub, 0xAC]), change_value))

    tx_outs.append((CScript([recipient_address, 0xAC]), amount_satoshis))

    raw_tx = rpc.create_raw_transaction(tx_ins, tx_outs)

    for i, (script_pub_key, _) in enumerate(tx_outs):
        sig = rpc.sign_raw_transaction_with_wallet(raw_tx, n_in=i, script_pub_key=script_pub_key, prev_pub_key=sender_secret.pub)
        raw_tx = sig['hex']

    txid = rpc.send_raw_transaction(raw_tx)
    return b2lx(txid)
