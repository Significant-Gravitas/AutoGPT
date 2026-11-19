import { bearer } from "better-auth/plugins";
import type { BetterAuthPlugin } from "better-auth";

export function nativeSessionBearer() {
  const plugin = bearer({ requireSignature: true });
  return {
    ...plugin,
    hooks: {
      ...plugin.hooks,
      after: plugin.hooks.after.map((hook) => ({
        ...hook,
        matcher(context: Parameters<typeof hook.matcher>[0]) {
          const headers = context.request?.headers ?? context.headers;
          return Boolean(
            headers?.get("X-AutoGPT-Native") === "1" &&
              ![
                "origin",
                "cookie",
                "sec-fetch-mode",
                "sec-fetch-site",
                "sec-fetch-dest",
                "sec-fetch-user",
              ].some((header) => headers.has(header)) &&
              hook.matcher(context),
          );
        },
      })),
    },
  } satisfies BetterAuthPlugin;
}
