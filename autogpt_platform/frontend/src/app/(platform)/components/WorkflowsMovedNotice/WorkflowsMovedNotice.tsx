"use client";

import { WorkflowsMovedDialog } from "./WorkflowsMovedDialog";
import { useWorkflowsMovedNotice } from "./useWorkflowsMovedNotice";

export function WorkflowsMovedNotice() {
  const { isOpen, dismiss } = useWorkflowsMovedNotice();
  return <WorkflowsMovedDialog isOpen={isOpen} onDismiss={dismiss} />;
}
