import React from "react";
import { Metadata } from "next/types";
import { redirect } from "next/navigation";
import BackendAPI from "@/lib/autogpt-server-api";
import { ProfileInfoForm } from "@/components/agptui/ProfileInfoForm";
import * as Sentry from "@sentry/nextjs";

// Force dynamic rendering to avoid static generation issues with cookies
export const dynamic = "force-dynamic";

export const metadata: Metadata = { title: "Profile - AutoGPT Platform" };

export default async function UserProfilePage(): Promise<React.ReactElement> {
  const api = new BackendAPI();
  const profile = await api.getStoreProfile().catch((error) => {
    Sentry.captureException(error);
    return null;
  });

  if (!profile) {
    redirect("/login");
  }

  return (
    <div className="flex flex-col items-center justify-center px-4">
      <ProfileInfoForm profile={profile} />
    </div>
  );
}
