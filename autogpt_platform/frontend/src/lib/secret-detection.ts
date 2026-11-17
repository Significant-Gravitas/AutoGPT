// Formats with a fixed prefix only: a bare long random string could as well be
// a commit SHA, a UUID or a file hash, and a warning on those would teach
// people to ignore it.
const SECRET_PATTERNS: RegExp[] = [
  // OpenAI project, service-account and admin keys, and Anthropic keys: the
  // prefix alone is distinctive, so any 20 key characters after it count.
  /\bsk-(?:proj|svcacct|admin|ant-[a-z]+\d+)-[A-Za-z0-9_-]{20,}/g,
  // GitHub classic and app tokens, including the stateless app installation
  // tokens (ghs_<app id>_<JWT>), then fine-grained tokens.
  /\bgh[pousr]_[A-Za-z0-9._-]{36,}/g,
  /\bgithub_pat_[A-Za-z0-9_]{22,}/g,
  // AWS access key IDs, long-lived (AKIA) and temporary (ASIA).
  /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g,
  // Slack bot, user, refresh and app-level tokens (xoxb-, xoxp-, xoxe-,
  // xapp-, and the rotating xoxe.xoxb- / xoxe.xapp- forms).
  /\b(?:xox[abposre]|xapp)-\d+-[A-Za-z0-9-]{10,}/g,
  // Stripe live secret and restricted keys.
  /\b[sr]k_live_[A-Za-z0-9]{20,}/g,
  // Google API keys.
  /\bAIza[A-Za-z0-9_-]{35}/g,
  // PEM private keys: RSA, EC, OPENSSH, ENCRYPTED, PGP ... PRIVATE KEY.
  /-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----/g,
];

// Other sk- keys: the run of 20 letters and digits after the prefix keeps
// hyphenated words like "sk-learn-and-pandas-..." out. Checked in two steps,
// since a single regex backtracks quadratically on a long run of "sk-".
const OTHER_SK_KEY = /\bsk-[A-Za-z0-9_-]+/g;
const TWENTY_LETTERS_OR_DIGITS = /[A-Za-z0-9]{20}/;

// Placeholders and documentation keys ("sk-xxxxxxxx...", AWS's
// "AKIAIOSFODNN7EXAMPLE") are not secrets, and pasting a .env.example should
// not set off the warning.
const REPEATED_CHARACTER = /([A-Za-z0-9])\1{7}/;

function isPlaceholder(candidate: string): boolean {
  return REPEATED_CHARACTER.test(candidate) || candidate.includes("EXAMPLE");
}

function* findCandidates(text: string): Generator<string> {
  for (const pattern of SECRET_PATTERNS) {
    for (const match of text.matchAll(pattern)) yield match[0];
  }
  for (const match of text.matchAll(OTHER_SK_KEY)) {
    if (TWENTY_LETTERS_OR_DIGITS.test(match[0].slice(3))) yield match[0];
  }
}

/** True when the text holds something shaped like a known API key, token or
 *  private key. */
export function containsLikelySecret(text: string): boolean {
  for (const candidate of findCandidates(text)) {
    if (!isPlaceholder(candidate)) return true;
  }
  return false;
}
