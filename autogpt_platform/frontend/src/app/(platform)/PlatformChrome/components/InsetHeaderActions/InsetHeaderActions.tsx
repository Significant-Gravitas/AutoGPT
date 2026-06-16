"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { Wallet } from "@/components/layout/Navbar/components/Wallet/Wallet";
import { getAccountMenuItems } from "@/components/layout/Navbar/helpers";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export function InsetHeaderActions() {
  const { user, isLoggedIn, isUserLoading } = useSupabase();
  const logoutInProgress = isLogoutInProgress();
  const dynamicMenuItems = getAccountMenuItems(user?.role);

  const { data: profile, isLoading: isProfileLoading } = useGetV2GetUserProfile({
    query: {
      select: okData,
      enabled: isLoggedIn && !!user && !logoutInProgress,
      queryKey: ["/api/store/profile", user?.id],
    },
  });

  if (!isLoggedIn) return null;

  const isLoadingProfile = isProfileLoading || isUserLoading;

  return (
    <div className="flex items-center gap-4">
      {profile && <Wallet key={profile.username} compact />}
      <AccountMenu
        userName={profile?.name || profile?.username}
        userEmail={user?.email}
        avatarSrc={profile?.avatar_url ?? ""}
        menuItemGroups={dynamicMenuItems}
        isLoading={isLoadingProfile}
      />
    </div>
  );
}
