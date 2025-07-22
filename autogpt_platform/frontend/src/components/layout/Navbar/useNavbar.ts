import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export function useNavbar() {
  const { isLoggedIn, isUserLoading } = useSupabase();

  console.log("isLoggedIn", isLoggedIn);

  const {
    data: profileResponse,
    isLoading: isProfileLoading,
    error: profileError,
  } = useGetV2GetUserProfile({
    query: {
      enabled: isLoggedIn === true,
    },
  });

  const profile = profileResponse?.data || null;
  const isLoading = isUserLoading || (isLoggedIn && isProfileLoading);

  return {
    isLoggedIn,
    profile,
    isLoading,
    profileError,
  };
}
