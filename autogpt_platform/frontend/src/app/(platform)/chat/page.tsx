"use client";

import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { CopilotPage } from "../copilot/CopilotPage";
import { RedirectToHome } from "./components/RedirectToHome/RedirectToHome";

export default function ChatPage() {
  const { enabled, ready } = useFlagStatus(Flag.TASK_BOARD);
  if (!ready) return null;
  return enabled ? <CopilotPage /> : <RedirectToHome />;
}
