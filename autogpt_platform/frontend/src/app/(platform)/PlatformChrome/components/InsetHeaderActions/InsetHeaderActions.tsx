"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { AgentActivityDropdown } from "@/components/layout/Navbar/components/AgentActivityDropdown/AgentActivityDropdown";
import { Wallet } from "@/components/layout/Navbar/components/Wallet/Wallet";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { UsageIndicator } from "@/app/(platform)/PlatformChrome/components/UsageIndicator/UsageIndicator";

export function InsetHeaderActions() {
  const { user, isLoggedIn } = useSupabase();
  const logoutInProgress = isLogoutInProgress();

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: okData,
      enabled: isLoggedIn && !!user && !logoutInProgress,
      queryKey: ["/api/store/profile", user?.id],
    },
  });

  if (!isLoggedIn) return null;

  return (
    <div className="flex items-center gap-3">
      <div className="[&_button:hover]:bg-zinc-200 [&_button]:flex [&_button]:h-7 [&_button]:w-7 [&_button]:items-center [&_button]:justify-center [&_button]:rounded-xl [&_button]:border [&_button]:border-zinc-200 [&_button]:bg-zinc-100 [&_button]:p-0 [&_svg]:!size-4">
        <AgentActivityDropdown />
      </div>
      <UsageIndicator />
      {profile && (
        <div className="[&_svg]:!size-4">
          <Wallet key={profile.username} compact />
        </div>
      )}
    </div>
  );
}
