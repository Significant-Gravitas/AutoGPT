import type { BetterAuthPlugin } from "better-auth";
import { apple } from "better-auth/social-providers";
import {
  appleWebCredentials,
  createAppleClientSecret,
} from "./apple-client-secret";

type AppleProvider = ReturnType<typeof apple>;

export function appleOAuth() {
  const credentials = appleWebCredentials();
  if (!credentials) return [];
  const web = credentials;
  return [
    {
      id: "apple-oauth",
      init(context) {
        return {
          context: {
            socialProviders: context.socialProviders.map((provider) => {
              if (provider.id !== "apple") return provider;
              // Better Auth caches provider options; sign again when each
              // outbound operation runs, even after a long consent flow.
              async function freshProvider() {
                return apple({
                  clientId: web.clientID,
                  clientSecret: await createAppleClientSecret(web),
                });
              }
              return {
                ...provider,
                async createAuthorizationURL(
                  input: Parameters<AppleProvider["createAuthorizationURL"]>[0],
                ) {
                  return (await freshProvider()).createAuthorizationURL(input);
                },
                async validateAuthorizationCode(
                  input: Parameters<
                    AppleProvider["validateAuthorizationCode"]
                  >[0],
                ) {
                  return (await freshProvider()).validateAuthorizationCode(
                    input,
                  );
                },
                async refreshAccessToken(token: string) {
                  return (await freshProvider()).refreshAccessToken(token);
                },
              };
            }),
          },
        };
      },
    } satisfies BetterAuthPlugin,
  ];
}
