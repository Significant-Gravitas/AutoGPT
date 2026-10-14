"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useParams } from "next/navigation";
import { Redirect } from "@/app/(platform)/home/components/Redirect/Redirect";
import { TaskBoardPage } from "@/app/(platform)/home/TaskBoardPage";

export default function BoardPage() {
  const { boardId } = useParams<{ boardId: string }>();
  const { enabled, ready } = useFlagStatus(Flag.TASK_BOARD);
  if (!ready) return <LoadingSpinner size="large" cover />;
  if (!enabled) return <Redirect href="/home" />;
  return <TaskBoardPage key={boardId} boardRef={boardId} />;
}
