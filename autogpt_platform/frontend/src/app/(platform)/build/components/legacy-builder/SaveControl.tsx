import React, { useEffect, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { GraphMeta } from "@/lib/autogpt-server-api";
import { Label } from "@/components/ui/label";
import { IconSave } from "@/components/ui/icons";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { getGetV2ListMySubmissionsQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import { CronExpressionDialog } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/components/cron-scheduler-dialog";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { CalendarClockIcon } from "lucide-react";

interface SaveControlProps {
  agentMeta: GraphMeta | null;
  agentName: string;
  agentDescription: string;
  agentRecommendedScheduleCron: string;
  canSave: boolean;
  onSave: () => Promise<void>;
  onNameChange: (name: string) => void;
  onDescriptionChange: (description: string) => void;
  onRecommendedScheduleCronChange: (cron: string) => void;
  pinSavePopover: boolean;
}

/**
 * A SaveControl component to be used within the ControlPanel. It allows the user to save the agent.
 * @param {Object} SaveControlProps - The properties of the SaveControl component.
 * @param {GraphMeta | null} SaveControlProps.agentMeta - The agent's metadata, or null if creating a new agent.
 * @param {string} SaveControlProps.agentName - The agent's name.
 * @param {string} SaveControlProps.agentDescription - The agent's description.
 * @param {boolean} SaveControlProps.canSave - Whether the button to save the agent should be enabled.
 * @param {() => void} SaveControlProps.onSave - Function to save the agent.
 * @param {(name: string) => void} SaveControlProps.onNameChange - Function to handle name changes.
 * @param {(description: string) => void} SaveControlProps.onDescriptionChange - Function to handle description changes.
 * @returns The SaveControl component.
 */
export const SaveControl = ({
  agentMeta,
  canSave,
  onSave,
  agentName,
  onNameChange,
  agentDescription,
  onDescriptionChange,
  agentRecommendedScheduleCron,
  onRecommendedScheduleCronChange,
  pinSavePopover,
}: SaveControlProps) => {
  /**
   * Note for improvement:
   * At the moment we are leveraging onDescriptionChange and onNameChange to handle the changes in the description and name of the agent.
   * We should migrate this to be handled with form controls and a form library.
   */

  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [cronScheduleDialogOpen, setCronScheduleDialogOpen] = useState(false);

  const handleScheduleChange = (cronExpression: string) => {
    onRecommendedScheduleCronChange(cronExpression);
  };

  useEffect(() => {
    const handleKeyDown = async (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "s") {
        event.preventDefault(); // Stop the browser default action
        await onSave(); // Call your save function
        queryClient.invalidateQueries({
          queryKey: getGetV2ListMySubmissionsQueryKey(),
        });
        toast({
          duration: 2000,
          title: "All changes saved successfully!",
        });
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onSave, toast]);

  return (
    <Popover open={pinSavePopover ? true : undefined}>
      <Tooltip delayDuration={500}>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              data-id="save-control-popover-trigger"
              data-testid="blocks-control-save-button"
              name="Save"
            >
              <IconSave className="dark:text-gray-300" />
            </Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Save</TooltipContent>
      </Tooltip>
      <PopoverContent
        side="right"
        sideOffset={15}
        align="start"
        data-id="save-control-popover-content"
        className="w-96 max-w-[400px]"
      >
        <Card className="border-none shadow-none dark:bg-slate-900">
          <CardContent className="p-4">
            <div className="space-y-3">
              <div>
                <Label htmlFor="name" className="dark:text-gray-300">
                  Name
                </Label>
                <Input
                  id="name"
                  placeholder="Enter your agent name"
                  value={agentName}
                  onChange={(e) => onNameChange(e.target.value)}
                  data-id="save-control-name-input"
                  data-testid="save-control-name-input"
                  maxLength={100}
                  className="mt-1"
                />
              </div>

              <div>
                <Label htmlFor="description" className="dark:text-gray-300">
                  Description
                </Label>
                <Input
                  id="description"
                  placeholder="Your agent description"
                  value={agentDescription}
                  onChange={(e) => onDescriptionChange(e.target.value)}
                  data-id="save-control-description-input"
                  data-testid="save-control-description-input"
                  maxLength={500}
                  className="mt-1"
                />
              </div>

              <div>
                <Label className="dark:text-gray-300">
                  Recommended Schedule
                </Label>
                <Button
                  variant="outline"
                  onClick={() => setCronScheduleDialogOpen(true)}
                  className="mt-1 w-full min-w-0 justify-start text-sm"
                  data-id="save-control-recommended-schedule-button"
                  data-testid="save-control-recommended-schedule-button"
                >
                  <CalendarClockIcon className="mr-2 h-4 w-4 flex-shrink-0" />
                  <span className="min-w-0 flex-1 truncate">
                    {agentRecommendedScheduleCron
                      ? humanizeCronExpression(agentRecommendedScheduleCron)
                      : "Set schedule"}
                  </span>
                </Button>
              </div>

              {agentMeta?.version && (
                <div>
                  <Label htmlFor="version" className="dark:text-gray-300">
                    Version
                  </Label>
                  <Input
                    id="version"
                    placeholder="Version"
                    value={agentMeta?.version || "-"}
                    disabled
                    data-testid="save-control-version-output"
                    className="mt-1"
                  />
                </div>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex flex-col items-stretch gap-2">
            <Button
              className="w-full dark:bg-slate-700 dark:text-slate-100 dark:hover:bg-slate-800"
              onClick={onSave}
              data-id="save-control-save-agent"
              data-testid="save-control-save-agent-button"
              disabled={!canSave}
            >
              Save Agent
            </Button>
          </CardFooter>
        </Card>
      </PopoverContent>
      <CronExpressionDialog
        open={cronScheduleDialogOpen}
        setOpen={setCronScheduleDialogOpen}
        onSubmit={handleScheduleChange}
        defaultCronExpression={agentRecommendedScheduleCron}
        title="Recommended Schedule"
      />
    </Popover>
  );
};
