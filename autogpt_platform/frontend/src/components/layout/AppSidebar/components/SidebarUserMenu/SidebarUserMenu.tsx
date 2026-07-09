"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { getAccountMenuItems } from "@/components/layout/Navbar/helpers";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export function SidebarUserMenu() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const logoutInProgress = isLogoutInProgress();
  const dynamicMenuItems = getAccountMenuItems(user?.role);

  const { data: profile, isLoading: isProfileLoading } = useGetV2GetUserProfile(
    {
      query: {
        select: okData,
        enabled: isLoggedIn && !!user && !logoutInProgress,
        queryKey: ["/api/store/profile", user?.id],
      },
    },
  );

  if (!isLoggedIn) return null;

  return (
    <AccountMenu
      userName={profile?.name || profile?.username}
      userEmail={user?.email}
      avatarSrc={profile?.avatar_url ?? ""}
      menuItemGroups={dynamicMenuItems}
      isLoading={isProfileLoading || isUserLoading}
    />
  );
}
