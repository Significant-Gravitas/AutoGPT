export type LayoutHint = "new" | "classic";

// Remembers which shell the layout flag last resolved to, so the server can
// render the right one for a returning user before the flag vendor answers.
// Functional, not tracking: same class as the sidebar's `sidebar_state`.
export const LAYOUT_HINT_COOKIE = "autogpt_layout";
const LAYOUT_HINT_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;
const SESSION_COOKIE_SUFFIX = "better-auth.session_token";

interface CookieEntry {
  name: string;
  value: string;
}

export function parseLayoutHint(
  raw: string | undefined,
): LayoutHint | undefined {
  return raw === "new" || raw === "classic" ? raw : undefined;
}

// The hint is only honoured alongside a session cookie: the new shell is a
// logged-in product, and a stale hint left behind by a logout must not paint
// the app sidebar for a visitor who will end up with the tour sidebar.
export function readLayoutHint(cookies: CookieEntry[]): LayoutHint | undefined {
  const hasSession = cookies.some((cookie) =>
    cookie.name.endsWith(SESSION_COOKIE_SUFFIX),
  );
  if (!hasSession) return undefined;
  const entry = cookies.find((cookie) => cookie.name === LAYOUT_HINT_COOKIE);
  return parseLayoutHint(entry?.value);
}

export function persistLayoutHint(isNewLayout: boolean) {
  if (typeof document === "undefined") return;
  const hint: LayoutHint = isNewLayout ? "new" : "classic";
  document.cookie = `${LAYOUT_HINT_COOKIE}=${hint}; path=/; max-age=${LAYOUT_HINT_MAX_AGE_SECONDS}; SameSite=Lax`;
}

interface ResolveLayoutInput {
  enabled: boolean;
  answered: boolean;
  ready: boolean;
  hint: LayoutHint | undefined;
}

// `undefined` means nothing trustworthy is known yet, so the chrome renders a
// neutral frame rather than guessing. A vendor answer always wins; the hint
// stands in until then and survives a vendor timeout, since the default it
// would otherwise fall back to is the shell the user most likely left.
export function resolveNewLayout({
  enabled,
  answered,
  ready,
  hint,
}: ResolveLayoutInput): boolean | undefined {
  if (answered) return enabled;
  if (hint !== undefined) return hint === "new";
  if (ready) return enabled;
  return undefined;
}
