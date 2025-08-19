import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useState } from "react";

export type RunVariant =
  | "manual"
  | "schedule"
  | "automatic-trigger"
  | "manual-trigger";

export function useAgentRunModal(agent: LibraryAgent) {
  const [isOpen, setIsOpen] = useState(false);
  const [showScheduleView, setShowScheduleView] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [scheduleName, setScheduleName] = useState("");
  const [cronExpression, setCronExpression] = useState("0 9 * * 1");

  // Determine the default run type based on agent capabilities
  const defaultRunType = agent.has_external_trigger
    ? "automatic-trigger"
    : "manual";

  function handleRun() {
    console.log("Running agent with inputs:", inputValues);
    setIsOpen(false);
  }

  function handleSchedule() {
    console.log("Creating schedule:", {
      scheduleName,
      cronExpression,
      inputValues,
    });
    setIsOpen(false);
  }

  function handleShowSchedule() {
    setShowScheduleView(true);
  }

  function handleGoBack() {
    setShowScheduleView(false);
  }

  function handleSetScheduleName(name: string) {
    setScheduleName(name);
  }

  function handleSetCronExpression(expression: string) {
    setCronExpression(expression);
  }

  return {
    isOpen,
    setIsOpen,
    showScheduleView,
    defaultRunType,
    inputValues,
    setInputValues,
    scheduleName,
    cronExpression,
    handleRun,
    handleSchedule,
    handleShowSchedule,
    handleGoBack,
    handleSetScheduleName,
    handleSetCronExpression,
  };
}
