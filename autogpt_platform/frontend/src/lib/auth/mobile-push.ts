import type { BetterAuthPlugin } from "better-auth";
import {
  APIError,
  createAuthEndpoint,
  getAuthoritativeSessionFromCtx,
} from "better-auth/api";
import { z } from "zod";
import { isMobileAuthSessionBlocked } from "./mobile-auth-helpers";

const registration = z
  .object({
    binding_id: z.string().uuid(),
    expected_user_id: z.string().min(1).max(128),
    provider: z.enum(["apns", "fcm"]),
    token: z
      .string()
      .min(32)
      .max(4096)
      .regex(/^[A-Za-z0-9_:\-]+$/),
    environment: z.enum(["sandbox", "production"]),
  })
  .strict()
  .refine((value) =>
    value.provider === "apns"
      ? /^[a-fA-F0-9]{32,512}$/.test(value.token)
      : value.environment === "production",
  );

export type MobilePushRegistration = z.infer<typeof registration> & {
  sessionID: string;
  origin: string;
};

export interface MobilePushStore {
  save: (registration: MobilePushRegistration) => Promise<void>;
  remove: (sessionID: string, bindingID?: string) => Promise<void>;
}

export function mobilePush(store: MobilePushStore) {
  return {
    id: "mobile-push",
    rateLimit: [
      {
        pathMatcher: (path) => path.startsWith("/mobile/push"),
        window: 60,
        max: 30,
      },
    ],
    endpoints: {
      registerMobilePush: createAuthEndpoint(
        "/mobile/push",
        { method: "POST", body: registration },
        async (ctx) => {
          const origin = new URL(ctx.context.baseURL).origin;
          if (ctx.headers?.get("origin") !== origin)
            throw new APIError("FORBIDDEN");
          const current = await getAuthoritativeSessionFromCtx(ctx);
          if (!current) throw new APIError("UNAUTHORIZED");
          if (current.user.id !== ctx.body.expected_user_id)
            throw new APIError("FORBIDDEN");
          if (
            isMobileAuthSessionBlocked(
              current,
              ctx.context.options.emailAndPassword?.requireEmailVerification,
            )
          ) {
            throw new APIError("FORBIDDEN");
          }
          await store.save({
            ...ctx.body,
            sessionID: current.session.id,
            origin,
          });
          ctx.setHeader("Cache-Control", "no-store");
          return ctx.json({ success: true });
        },
      ),
      removeMobilePush: createAuthEndpoint(
        "/mobile/push/remove",
        {
          method: "POST",
          body: z.object({ binding_id: z.string().uuid().optional() }).strict(),
        },
        async (ctx) => {
          if (
            ctx.headers?.get("origin") !== new URL(ctx.context.baseURL).origin
          )
            throw new APIError("FORBIDDEN");
          const current = await getAuthoritativeSessionFromCtx(ctx);
          if (!current) throw new APIError("UNAUTHORIZED");
          await store.remove(current.session.id, ctx.body.binding_id);
          ctx.setHeader("Cache-Control", "no-store");
          return ctx.json({ success: true });
        },
      ),
    },
  } satisfies BetterAuthPlugin;
}
