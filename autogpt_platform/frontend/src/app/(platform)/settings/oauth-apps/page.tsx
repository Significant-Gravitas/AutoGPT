"use client";

import { OAuthAppsHeader } from "./components/OAuthAppsHeader/OAuthAppsHeader";
import { OAuthGrantList } from "./components/OAuthGrantList/OAuthGrantList";

export default function SettingsOAuthAppsPage() {
  return (
    <>
      <OAuthAppsHeader />
      <OAuthGrantList />
    </>
  );
}
