"use client";

import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { CopilotPage } from "../copilot/CopilotPage";
import { TaskBoardPage } from "./TaskBoardPage";

export default function HomePage() {
  const { enabled, ready } = useFlagStatus(Flag.TASK_BOARD);
  return ready && enabled ? <TaskBoardPage /> : <CopilotPage />;
}
