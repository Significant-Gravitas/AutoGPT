"use client";

import { BotConnectionNotice } from "./components/BotConnectionNotice/BotConnectionNotice";
import { BotsHeader } from "./components/BotsHeader/BotsHeader";
import { BotsList } from "./components/BotsList/BotsList";

export default function SettingsBotsPage() {
  return (
    <>
      <BotsHeader />
      <BotConnectionNotice />
      <BotsList />
    </>
  );
}
