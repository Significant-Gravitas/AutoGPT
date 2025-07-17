import { prefetchGetV2GetUserProfileQuery } from "@/app/api/__generated__/endpoints/store/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { getServerUser } from "@/lib/supabase/server/getServerUser";

export async function getNavbarAccountData() {
  const { user } = await getServerUser();
  const isLoggedIn = Boolean(user);
  const queryClient = getQueryClient();

  if (!isLoggedIn) {
    return {
      profile: null,
      isLoggedIn,
    };
  }
  try {
    await prefetchGetV2GetUserProfileQuery(queryClient);
  } catch (error) {
    console.error("Error fetching profile:", error);
  }

  return {
    isLoggedIn,
  };
}
