import * as React from "react";
import { Metadata } from "next/types";
import { ProfileInfoForm } from "@/components/agptui/ProfileInfoForm";
import BackendAPI from "@/lib/autogpt-server-api";
import { CreatorDetails } from "@/lib/autogpt-server-api/types";

export const metadata: Metadata = { title: "Profile - AutoGPT Platform" };

export default async function UserProfilePage({}: {}) {
  const api = new BackendAPI();
  const profile = await api.getStoreProfile().catch((error) => {
    console.error("Error fetching profile:", error);
    return null;
  });

  if (!profile) {
    return (
      <div className="flex flex-col items-center justify-center p-4">
        <p>Please log in to view your profile</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center px-4">
      <ProfileInfoForm profile={profile} />
    </div>
  );
}
