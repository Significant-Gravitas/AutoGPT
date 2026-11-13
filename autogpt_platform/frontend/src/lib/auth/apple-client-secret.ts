import { importPKCS8, SignJWT } from "jose";
import { APIError } from "better-auth/api";

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
  try {
    const key = await importPKCS8(
      credentials.privateKey.replaceAll("\\n", "\n"),
      "ES256",
    );
    const now = Math.floor(Date.now() / 1000);
    return await new SignJWT({})
      .setProtectedHeader({ alg: "ES256", kid: credentials.keyID })
      .setIssuer(credentials.teamID)
      .setSubject(credentials.clientID)
      .setAudience("https://appleid.apple.com")
      .setIssuedAt(now)
      .setExpirationTime(now + 300)
      .sign(key);
  } catch {
    console.warn(
      "Apple client-secret signing failed. Check the Apple key configuration.",
    );
    throw new APIError("SERVICE_UNAVAILABLE", {
      code: "APPLE_SIGN_IN_UNAVAILABLE",
      message: "Sign in with Apple is temporarily unavailable.",
    });
  }
}
