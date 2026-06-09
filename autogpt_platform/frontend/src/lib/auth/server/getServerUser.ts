import { getServerSupabase } from "./getServerSupabase";

export async function getServerUser() {
  const authClient = await getServerSupabase();
  const {
    data: { user },
    error,
  } = await authClient.auth.getUser();

  if (error || !user) {
    return {
      user: null,
      role: null,
      error: error?.message || "No user found in the response",
    };
  }

  return {
    user,
    role: user.role || null,
    error: null,
  };
}
