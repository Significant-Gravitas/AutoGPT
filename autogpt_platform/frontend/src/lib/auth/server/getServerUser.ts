import { mapSessionUser, type User } from "@/lib/auth/types";
import { getServerSession } from "./getServerSession";

export async function getServerUser(): Promise<{
  user: User | null;
  role: string | null;
  error: string | null;
}> {
  try {
    const session = await getServerSession();

    if (!session?.user) {
      return { user: null, role: null, error: "No user found in the response" };
    }

    const user = mapSessionUser(session.user);
    return { user, role: user.role || null, error: null };
  } catch (error) {
    console.error("Unexpected error in getServerUser:", error);
    return {
      user: null,
      role: null,
      error: `Unexpected error: ${(error as Error).message}`,
    };
  }
}
