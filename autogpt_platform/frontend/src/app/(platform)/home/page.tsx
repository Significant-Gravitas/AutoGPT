"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { CopilotPage } from "../copilot/CopilotPage";
import { Redirect } from "./components/Redirect/Redirect";
import { DEFAULT_BOARD_REF } from "./board-helpers";
import { TaskBoardPage } from "./TaskBoardPage";
import { useChatLinkTarget } from "./useChatLinkTarget";

export default function HomePage() {
  const { enabled, ready } = useFlagStatus(Flag.TASK_BOARD);
  const chatLink = useChatLinkTarget();
  if (!ready || !enabled) return <CopilotPage />;
  if (chatLink === undefined) return <LoadingSpinner size="large" cover />;
  if (chatLink) return <Redirect href={chatLink} />;
  return <TaskBoardPage boardRef={DEFAULT_BOARD_REF} />;
}
