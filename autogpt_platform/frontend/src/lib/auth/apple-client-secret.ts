import { importPKCS8, SignJWT } from "jose";

export function appleWebCredentials() {
  const clientID = process.env.AUTH_APPLE_CLIENT_ID?.trim();
  const teamID = process.env.AUTH_APPLE_TEAM_ID?.trim();
  const keyID = process.env.AUTH_APPLE_KEY_ID?.trim();
  const privateKey = process.env.AUTH_APPLE_PRIVATE_KEY?.trim();
  if (!clientID || !teamID || !keyID || !privateKey) return null;
  return { clientID, teamID, keyID, privateKey };
}

export async function createAppleClientSecret(
  credentials: NonNullable<ReturnType<typeof appleWebCredentials>>,
) {
  const key = await importPKCS8(
    credentials.privateKey.replaceAll("\\n", "\n"),
    "ES256",
  );
  const now = Math.floor(Date.now() / 1000);
  return new SignJWT({})
    .setProtectedHeader({ alg: "ES256", kid: credentials.keyID })
    .setIssuer(credentials.teamID)
    .setSubject(credentials.clientID)
    .setAudience("https://appleid.apple.com")
    .setIssuedAt(now)
    .setExpirationTime(now + 300)
    .sign(key);
}
