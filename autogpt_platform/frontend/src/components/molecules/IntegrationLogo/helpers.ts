/** Map a provider slug to its logo in `public/integrations`.
 *
 * The slug is scrubbed rather than trusted: provider strings reach this from
 * tool payloads and credential rows, and an unscrubbed one would let `../../`
 * walk out of the integrations directory. A slug that scrubs to nothing has no
 * icon at all, which callers render as their fallback.
 */
export function integrationIconSrc(provider: string): string | null {
  const slug = provider
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_")
    .replace(/[^a-z0-9_]/g, "");
  return slug ? `/integrations/${slug}.png` : null;
}
