import { environment } from "@/services/environment";

/** Dev-only: local/dev environments, and NEXT_PUBLIC_TOKEN_DEVTOOL can turn
 *  it off explicitly (unset = on). */
export function isTokenDevtoolEnabled(): boolean {
  if (process.env.NEXT_PUBLIC_TOKEN_DEVTOOL === "false") return false;
  return (
    environment.isDevelopmentBuild() ||
    environment.isLocal() ||
    environment.isDev()
  );
}
