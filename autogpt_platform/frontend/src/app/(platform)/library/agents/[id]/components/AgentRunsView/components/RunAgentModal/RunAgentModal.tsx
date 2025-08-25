"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useAgentRunModal } from "./useAgentRunModal";
import { ModalHeader } from "./components/ModalHeader/ModalHeader";

interface Props {
  triggerSlot: React.ReactNode;
  agent: LibraryAgent;
  agentId: string;
  agentVersion?: number;
}

export function RunAgentModal({ triggerSlot, agent }: Props) {
  const {
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
  } = useAgentRunModal(agent);

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "600px", maxHeight: "90vh" }}
    >
      <Dialog.Trigger>{triggerSlot}</Dialog.Trigger>
      <Dialog.Content>
        <div className="space-y-6">
          <ModalHeader showScheduleView={showScheduleView} agent={agent} />

          {/* Content */}
          <div className="space-y-6">
            {!showScheduleView ? (
              /* Default Run View */
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-neutral-800">
                  {defaultRunType === "automatic-trigger"
                    ? "Trigger Setup"
                    : "Agent Setup"}
                </h3>

                {defaultRunType === "automatic-trigger" && (
                  <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
                    <div className="flex items-start">
                      <div className="flex-shrink-0">
                        <svg
                          className="h-5 w-5 text-blue-400"
                          viewBox="0 0 20 20"
                          fill="currentColor"
                        >
                          <path
                            fillRule="evenodd"
                            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <h3 className="text-sm font-medium text-blue-800">
                          Webhook Trigger
                        </h3>
                        <div className="mt-2 text-sm text-blue-700">
                          <p>
                            This will create a webhook endpoint that
                            automatically runs your agent when triggered by
                            external events.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Input fields */}
                {agent.input_schema &&
                typeof agent.input_schema === "object" &&
                "properties" in agent.input_schema ? (
                  Object.entries(
                    (agent.input_schema as any).properties || {},
                  ).map(([key, schema]: [string, any]) => (
                    <Input
                      key={key}
                      id={key}
                      label={schema.title || key}
                      value={inputValues[key] || ""}
                      onChange={(e) =>
                        setInputValues((prev) => ({
                          ...prev,
                          [key]: e.target.value,
                        }))
                      }
                      placeholder={schema.description}
                    />
                  ))
                ) : (
                  <div className="rounded-lg bg-neutral-50 p-4 text-sm text-neutral-500">
                    No input fields required for this agent
                  </div>
                )}

                <div className="flex justify-end gap-3 border-t pt-4">
                  {!agent.has_external_trigger && (
                    <Button variant="secondary" onClick={handleShowSchedule}>
                      Schedule Run
                    </Button>
                  )}
                  <Button variant="primary" onClick={handleRun}>
                    {defaultRunType === "automatic-trigger"
                      ? "Set up Trigger"
                      : "Run Agent"}
                  </Button>
                </div>
              </div>
            ) : (
              /* Schedule View */
              <div className="space-y-4">
                <h3 className="text-lg font-medium text-neutral-800">
                  Schedule Setup
                </h3>

                <Input
                  id="schedule-name"
                  label="Schedule Name"
                  value={scheduleName}
                  onChange={(e) => handleSetScheduleName(e.target.value)}
                  placeholder="Enter a name for this schedule"
                />

                <Input
                  id="cron-expression"
                  label="Schedule Pattern"
                  value={cronExpression}
                  onChange={(e) => handleSetCronExpression(e.target.value)}
                  placeholder="0 9 * * 1"
                  hint={
                    <span className="text-xs text-neutral-500">
                      Format: minute hour day month weekday
                    </span>
                  }
                />

                {/* Input fields */}
                {agent.input_schema &&
                typeof agent.input_schema === "object" &&
                "properties" in agent.input_schema ? (
                  Object.entries(
                    (agent.input_schema as any).properties || {},
                  ).map(([key, schema]: [string, any]) => (
                    <Input
                      key={key}
                      id={key}
                      label={schema.title || key}
                      value={inputValues[key] || ""}
                      onChange={(e) =>
                        setInputValues((prev) => ({
                          ...prev,
                          [key]: e.target.value,
                        }))
                      }
                      placeholder={schema.description}
                    />
                  ))
                ) : (
                  <div className="rounded-lg bg-neutral-50 p-4 text-sm text-neutral-500">
                    No input fields required for this agent
                  </div>
                )}

                <div className="flex justify-end gap-3 border-t pt-4">
                  <Button variant="ghost" onClick={handleGoBack}>
                    Go Back
                  </Button>
                  <Button variant="primary" onClick={handleSchedule}>
                    Create Schedule
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
