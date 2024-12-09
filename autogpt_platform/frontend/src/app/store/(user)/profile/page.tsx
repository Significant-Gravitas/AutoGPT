import * as React from "react";
import { ProfileInfoForm } from "@/components/agptui/ProfileInfoForm";
import AutoGPTServerAPIServerSide from "@/lib/autogpt-server-api";
import { createServerClient } from "@/lib/supabase/server";
import { CreatorDetails } from "@/lib/autogpt-server-api/types";

async function getProfileData() {
  // Get the supabase client first
  const supabase = createServerClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session) {
    console.warn("--- No session found in profile page");
    return { profile: null };
  }

  // Create API client with the same supabase instance
  const api = new AutoGPTServerAPIServerSide(
    process.env.NEXT_PUBLIC_AGPT_SERVER_URL,
    process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL,
    supabase, // Pass the supabase client instance
  );

  try {
    const profile = await api.getStoreProfile("profile");
    return {
      profile,
    };
  } catch (error) {
    console.error("Error fetching profile:", error);
    return {
      profile: null,
    };
  }
}

export default async function Page({}: {}) {
  const { profile } = await getProfileData();

  if (!profile) {
    return (
      <div className="flex flex-col items-center justify-center p-4">
        <p>Please log in to view your profile</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center px-4">
      <ProfileInfoForm profile={profile as CreatorDetails} />
    </div>
  );
}
