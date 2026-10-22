/**
 * Experiment flag for the drawn-line thinking indicator.
 *
 * This is a build-time env flag, NOT a LaunchDarkly flag: the repo's feature
 * flags (`Flag` in `@/services/feature-flags/use-get-flag`) are declared on the
 * backend (`FrontendFlag`) and mirrored into the generated API client, which is
 * out of scope for a frontend-only experiment. To try it locally set
 *
 *   NEXT_PUBLIC_COPILOT_DRAWN_LINE_LOADER=true
 *
 * in `.env.local` and restart `pnpm dev` (NEXT_PUBLIC_* values are inlined at
 * build time). Anything other than "true" keeps the pixel-grid loader.
 *
 * The `process.env.NEXT_PUBLIC_*` access must stay a literal member lookup so
 * Next.js can inline it into the client bundle.
 */
export const DRAWN_LINE_LOADER_ENV_VAR =
  "NEXT_PUBLIC_COPILOT_DRAWN_LINE_LOADER";

export function isDrawnLineLoaderEnabled(): boolean {
  return process.env.NEXT_PUBLIC_COPILOT_DRAWN_LINE_LOADER === "true";
}
