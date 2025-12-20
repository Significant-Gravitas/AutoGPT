import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { useAuth } from "@/lib/auth";

export function useNavbar() {
  const { isLoggedIn, isUserLoading } = useAuth();

  const {
    data: profileResponse,
    isLoading: isProfileLoading,
    error: profileError,
  } = useGetV2GetUserProfile({
    query: {
      // Only fetch when user is confirmed logged in and auth loading is complete
      enabled: isLoggedIn === true && !isUserLoading,
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
