"use client";

import { useGetV2GetUserProfile } from "@/app/api/__generated__/endpoints/store/store";
import { ProfileInfoForm } from "@/components/__legacy__/ProfileInfoForm";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { isLogoutInProgress } from "@/lib/autogpt-server-api/helpers";
import { ProfileDetails } from "@/lib/autogpt-server-api/types";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { ProfileLoading } from "./ProfileLoading";

export default function UserProfilePage() {
  const { user } = useSupabase();
  const logoutInProgress = isLogoutInProgress();

  const {
    data: profile,
    isLoading,
    isError,
    error,
    refetch,
  } = useGetV2GetUserProfile<ProfileDetails | null>({
    query: {
      enabled: !!user && !logoutInProgress,
      select: (res) => {
        if (res.status === 200) {
          return {
            ...res.data,
            avatar_url: res.data.avatar_url ?? "",
          };
        }
        return null;
      },
    },
  });

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center px-4">
        <ErrorCard
          responseError={
            error
              ? {
                  detail: error.detail,
                }
              : undefined
          }
          context="profile"
          onRetry={() => {
            void refetch();
          }}
        />
      </div>
    );
  }

  if (isLoading || !user || !profile) {
    return <ProfileLoading />;
  }

  return (
    <div className="flex flex-col items-center justify-center px-4">
      <ProfileInfoForm profile={profile} />
    </div>
  );
}
