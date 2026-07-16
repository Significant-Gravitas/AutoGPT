"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { AgentActivityDropdown } from "@/components/layout/Navbar/components/AgentActivityDropdown/AgentActivityDropdown";
import { Wallet } from "@/components/layout/Navbar/components/Wallet/Wallet";
import { getAccountMenuItems } from "@/components/layout/Navbar/helpers";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { UsageIndicator } from "@/app/(platform)/PlatformChrome/components/UsageIndicator/UsageIndicator";

export function InsetHeaderActions() {
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

  const isLoadingProfile = isProfileLoading || isUserLoading;

  return (
    <div className="flex items-center gap-1 rounded-full border border-black/5 bg-white/60 p-1.5 shadow-[0_1px_2px_rgba(0,0,0,0.04),0_12px_32px_-8px_rgba(0,0,0,0.12)] backdrop-blur-xl">
      <div className="[&_button:hover]:bg-black/5 [&_button]:flex [&_button]:h-8 [&_button]:w-8 [&_button]:items-center [&_button]:justify-center [&_button]:rounded-full [&_button]:bg-transparent [&_button]:p-0 [&_button]:transition-colors [&_svg]:!size-5">
        <AgentActivityDropdown />
      </div>
      <UsageIndicator />
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
