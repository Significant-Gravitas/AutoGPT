import { getServerUser } from "@/lib/supabase/server/getServerUser";

export async function getNavbarAccountData() {
  const { user } = await getServerUser();
  const isLoggedIn = Boolean(user);

  if (!isLoggedIn) {
    return {
      profile: null,
      isLoggedIn,
    };
  }

  return {
    isLoggedIn,
  };
}
