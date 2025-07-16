import BackendAPI from "@/lib/autogpt-server-api";
import { getServerUser } from "@/lib/supabase/server/getServerUser";

export async function getNavbarAccountData() {
  const { user } = await getServerUser();
  const api = new BackendAPI();
  const isLoggedIn = Boolean(user);

  if (!isLoggedIn) {
    return {
      profile: null,
      isLoggedIn,
    };
  }

  let profile = null;

  try {
    profile = await api.getStoreProfile();
  } catch (error) {
    console.error("Error fetching profile:", error);
    profile = null;
  }

  return {
    profile,
    isLoggedIn,
  };
}
