import { environment } from "@/services/environment";

/** Dev-only. Requires a development build (`pnpm dev`) or the cloud dev
 *  deployment. Deliberately does NOT key off `environment.isLocal()`: that is
 *  the default branch of `getBehaveAs()`, so it is true for any
 *  NEXT_PUBLIC_BEHAVE_AS that is not exactly "CLOUD" — including the
 *  self-hosted single-container image, which builds with NODE_ENV=production
 *  and NEXT_PUBLIC_BEHAVE_AS=LOCAL. Gating on the build type keeps the badge
 *  and the SSE tap out of every production build.
 *
 *  NEXT_PUBLIC_TOKEN_DEVTOOL=false turns it off on top of that. Next inlines
 *  NEXT_PUBLIC_* at build time, so flipping it needs a rebuild. */
export function isTokenDevtoolEnabled(): boolean {
  if (isDisabled(process.env.NEXT_PUBLIC_TOKEN_DEVTOOL)) return false;
  return environment.isDevelopmentBuild() || environment.isDev();
}

function isDisabled(value: string | undefined): boolean {
  const normalized = value?.trim().toLowerCase();
  return normalized === "false" || normalized === "0" || normalized === "off";
}
