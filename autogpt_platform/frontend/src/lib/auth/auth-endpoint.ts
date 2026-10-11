import { getCurrentAuthContext } from "@better-auth/core/context";

/**
 * The Better Auth endpoint the current callback runs under, such as
 * "/sign-in/email" or "/change-email". Better Auth runs every endpoint inside
 * an async context, server-side auth.api calls included (they carry no
 * request), and the email callbacks it fires run inside it too. Outside one,
 * both are undefined.
 */
export async function currentAuthEndpoint(): Promise<{
  path?: string;
  body?: unknown;
}> {
  try {
    const { path, body } = await getCurrentAuthContext();
    return { path, body };
  } catch {
    return {};
  }
}
