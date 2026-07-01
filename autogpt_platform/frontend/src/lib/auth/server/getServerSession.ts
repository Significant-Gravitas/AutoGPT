import { auth } from "@/lib/auth/auth";
import { headers } from "next/headers";
import { cache } from "react";

/**
 * Reads the Better Auth session for the current request. React-cached so
 * repeated calls within one request hit the DB (or cookie cache) once.
 */
export const getServerSession = cache(async () => {
  try {
    return await auth.api.getSession({ headers: await headers() });
  } catch (error) {
    console.error("Failed to read auth session:", error);
    return null;
  }
});
