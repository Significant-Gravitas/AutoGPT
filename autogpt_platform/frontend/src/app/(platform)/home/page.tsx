"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { CopilotPage } from "../copilot/CopilotPage";
import { RedirectToChat } from "./components/RedirectToChat/RedirectToChat";
import { TaskBoardPage } from "./TaskBoardPage";
import { useChatLinkTarget } from "./useChatLinkTarget";

export default function HomePage() {
  const { enabled, ready } = useFlagStatus(Flag.TASK_BOARD);
  const chatLink = useChatLinkTarget();
  if (!ready || !enabled) return <CopilotPage />;
  if (chatLink === undefined) return <LoadingSpinner size="large" cover />;
  if (chatLink) return <RedirectToChat href={chatLink} />;
  return <TaskBoardPage />;
}
