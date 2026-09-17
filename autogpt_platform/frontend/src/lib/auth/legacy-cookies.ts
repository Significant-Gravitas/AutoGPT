/**
 * Whether the platform may consume (clear) a caller's legacy Supabase cookies.
 *
 * Clearing them is what stops the middleware redirecting to the bridge
 * forever, so the bridge clears on every path that reaches a verdict — but
 * not before it is capable of reaching one. Without `SUPABASE_JWT_SECRET`
 * verification can only fail, and clearing first would turn a missing env var
 * into permanent, unrecoverable session loss for every migrating user: the
 * cookie would be gone before anyone noticed the misconfiguration.
 *
 * The middleware gates on the same check before redirecting to the bridge at
 * all — with no secret, the bridge would bounce straight back to /login with
 * the cookies intact and the pair would loop forever. Shared module (with no
 * server-only imports) so the edge middleware and the bridge can't drift.
 */
export function canConsumeLegacyCookies(
  secret: string | undefined = process.env.SUPABASE_JWT_SECRET,
): boolean {
  return Boolean(secret);
}
