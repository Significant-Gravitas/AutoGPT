import {
  auth,
  getServerAuthSession,
  toAuthError,
  toLegacyAuthSession,
  toLegacyAuthUser,
} from "@/lib/auth/auth";
import { headers } from "next/headers";
import { cache } from "react";

async function getRequestHeaders() {
  return new Headers(await headers());
}

export const getServerSupabase = cache(async () => {
  return {
    auth: {
      async getUser() {
        try {
          const session = await getServerAuthSession();
          return {
            data: {
              user: session ? toLegacyAuthUser(session.user) : null,
            },
            error: null,
          };
        } catch (error) {
          return {
            data: { user: null },
            error: toAuthError(error),
          };
        }
      },
      async getSession() {
        try {
          const session = await getServerAuthSession();
          return {
            data: {
              session: session ? await toLegacyAuthSession(session) : null,
            },
            error: null,
          };
        } catch (error) {
          return {
            data: { session: null },
            error: toAuthError(error),
          };
        }
      },
      async signInWithPassword(data: { email: string; password: string }) {
        try {
          const result = await auth.api.signInEmail({
            body: data,
            headers: await getRequestHeaders(),
          });
          return {
            data: result,
            error: null,
          };
        } catch (error) {
          return {
            data: null,
            error: toAuthError(error),
          };
        }
      },
      async signUp(data: {
        email: string;
        password: string;
        name?: string;
        options?: {
          data?: Record<string, unknown>;
          emailRedirectTo?: string;
        };
      }) {
        try {
          const result = await auth.api.signUpEmail({
            body: {
              email: data.email,
              password: data.password,
              name: data.name || data.email.split("@")[0],
              callbackURL: data.options?.emailRedirectTo,
              ...(data.options?.data ?? {}),
            },
            headers: await getRequestHeaders(),
          });
          return {
            data: result,
            error: null,
          };
        } catch (error) {
          return {
            data: null,
            error: toAuthError(error),
          };
        }
      },
      async signInWithOAuth({
        provider,
        options,
      }: {
        provider: string;
        options?: { redirectTo?: string };
      }) {
        try {
          const result = await auth.api.signInSocial({
            body: {
              provider,
              callbackURL: options?.redirectTo,
              disableRedirect: true,
            },
            headers: await getRequestHeaders(),
          });
          return {
            data: result,
            error: null,
          };
        } catch (error) {
          return {
            data: null,
            error: toAuthError(error),
          };
        }
      },
      async resetPasswordForEmail(
        email: string,
        options?: { redirectTo?: string },
      ) {
        try {
          const result = await auth.api.requestPasswordReset({
            body: {
              email,
              redirectTo: options?.redirectTo,
            },
            headers: await getRequestHeaders(),
          });
          return {
            data: result,
            error: null,
          };
        } catch (error) {
          return {
            data: null,
            error: toAuthError(error),
          };
        }
      },
      async updateUser(payload: {
        email?: string;
        password?: string;
        data?: { full_name?: string };
      }) {
        try {
          if (payload.password) {
            throw new Error(
              "Password updates must use the dedicated reset-password flow",
            );
          }

          let emailResult: unknown = null;
          if (payload.email) {
            emailResult = await auth.api.changeEmail({
              body: {
                newEmail: payload.email,
              },
              headers: await getRequestHeaders(),
            });
          }

          const updateBody: Record<string, unknown> = {};
          if (payload.data?.full_name) {
            updateBody.name = payload.data.full_name;
          }

          const result =
            Object.keys(updateBody).length > 0
              ? await auth.api.updateUser({
                  body: updateBody,
                  headers: await getRequestHeaders(),
                })
              : emailResult;

          return {
            data: result,
            error: null,
          };
        } catch (error) {
          return {
            data: null,
            error: toAuthError(error),
          };
        }
      },
      async signOut() {
        try {
          const result = await auth.api.signOut({
            headers: await getRequestHeaders(),
          });

          return {
            data: result,
            error: null,
          };
        } catch (error) {
          return {
            data: null,
            error: toAuthError(error),
          };
        }
      },
      async refreshSession() {
        return this.getUser();
      },
      async setSession() {
        return {
          data: null,
          error: null,
        };
      },
    },
  };
});
