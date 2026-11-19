import {
  createHmac,
  randomBytes,
  randomInt,
  timingSafeEqual,
} from "node:crypto";
import { SIGN_IN_CODE_LENGTH } from "./sign-in-code";

const CODE_PATTERN = new RegExp(`^[0-9]{${SIGN_IN_CODE_LENGTH}}$`);

/** A waiting sign-in's id, which its browser holds in a signed cookie. */
export function newPendingID() {
  return randomBytes(32).toString("base64url");
}

export function newCode() {
  return randomInt(0, 10 ** SIGN_IN_CODE_LENGTH)
    .toString()
    .padStart(SIGN_IN_CODE_LENGTH, "0");
}

// Keyed by the sign-in's id as well, so a code means nothing for another one.
export function hashCode(secret: string, id: string, code: string) {
  return createHmac("sha256", secret).update(`${id}:${code}`).digest("hex");
}

export function codeMatches(
  secret: string,
  id: string,
  code: string,
  hash: string,
) {
  if (!CODE_PATTERN.test(code)) return false;
  const expected = Buffer.from(hash, "hex");
  const actual = Buffer.from(hashCode(secret, id, code), "hex");
  return expected.length === actual.length && timingSafeEqual(expected, actual);
}
