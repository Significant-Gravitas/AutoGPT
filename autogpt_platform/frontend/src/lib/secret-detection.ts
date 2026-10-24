// Formats with a fixed prefix only: a bare long random string could as well be
// a commit SHA, a UUID or a file hash, and a warning on those would teach
// people to ignore it.
const SECRET_PATTERNS: RegExp[] = [
  // OpenAI project, service-account and admin keys, and Anthropic keys: the
  // prefix alone is distinctive, so any 20 key characters after it count.
  /\bsk-(?:proj|svcacct|admin|ant-[a-z]+\d+)-[A-Za-z0-9_-]{20,}/,
  // Other sk- keys. The run of 20 letters and digits keeps hyphenated words
  // like "sk-learn-and-pandas-..." out.
  /\bsk-[A-Za-z0-9_-]*[A-Za-z0-9]{20}/,
  // GitHub classic and app tokens, including the stateless app installation
  // tokens (ghs_<app id>_<JWT>), then fine-grained tokens.
  /\bgh[pousr]_[A-Za-z0-9._-]{36,}/,
  /\bgithub_pat_[A-Za-z0-9_]{22,}/,
  // AWS access key IDs, long-lived (AKIA) and temporary (ASIA).
  /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/,
  // Slack bot, user, refresh and app-level tokens (xoxb-, xoxp-, xoxe-,
  // xapp-, and the rotating xoxe.xoxb- / xoxe.xapp- forms).
  /\b(?:xox[abposre]|xapp)-\d+-[A-Za-z0-9-]{10,}/,
  // Stripe live secret and restricted keys.
  /\b[sr]k_live_[A-Za-z0-9]{20,}/,
  // Google API keys.
  /\bAIza[A-Za-z0-9_-]{35}/,
  // PEM private keys: RSA, EC, OPENSSH, ENCRYPTED, PGP ... PRIVATE KEY.
  /-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----/,
];

/** True when the text holds something shaped like a known API key, token or
 *  private key. */
export function containsLikelySecret(text: string): boolean {
  return SECRET_PATTERNS.some((pattern) => pattern.test(text));
}
