import { getV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
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

  let profile = null;

  try {
    const profileResponse = await getV2GetUserProfile();
    profile = profileResponse.data || null;
  } catch (error) {
    console.error("Error fetching profile:", error);
    profile = null;
  }

  return {
    profile,
    isLoggedIn,
  };
}
