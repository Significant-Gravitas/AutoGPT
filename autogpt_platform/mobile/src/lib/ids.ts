import * as Crypto from "expo-crypto";

/** A fresh per-send idempotency key. Retries of the same send must reuse it. */
export function newMessageID(): string {
  return Crypto.randomUUID();
}

/** A short, keychain-safe digest used to partition stored data. */
export async function stableDigest(value: string): Promise<string> {
  const digest = await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    value,
    { encoding: Crypto.CryptoEncoding.HEX },
  );
  return digest.slice(0, 32);
}
