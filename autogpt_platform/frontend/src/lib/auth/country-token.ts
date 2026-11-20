import { mintServiceToken } from "./service-token";

export const CLIENT_COUNTRY_SCOPE = "client-country";

// Tokens live 60s; reusing one for 30s leaves every sent token 30s to spare
// while minting at most once per country per half-minute per instance.
const REUSE_MS = 30_000;

const minted = new Map<string, { token: string; mintedAt: number }>();

/**
 * The visitor's country as a frontend service token the backend can verify.
 * The backend is reachable directly, so a plain country header would be
 * whatever the caller typed; a signed one can only come from here.
 */
export async function getCountryToken(country: string) {
  const code = country.trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(code)) return null;
  const cached = minted.get(code);
  if (cached && Date.now() - cached.mintedAt < REUSE_MS) return cached.token;
  const token = await mintServiceToken(CLIENT_COUNTRY_SCOPE, { country: code });
  minted.set(code, { token, mintedAt: Date.now() });
  return token;
}
