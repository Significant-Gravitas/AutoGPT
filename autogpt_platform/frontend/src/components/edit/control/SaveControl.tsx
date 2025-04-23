import React, { useCallback, useEffect } from "react";
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
} from "@/components/ui/tooltip";
import { useToast } from "@/components/ui/use-toast";

interface SaveControlProps {
  agentMeta: GraphMeta | null;
  agentName: string;
  agentDescription: string;
  canSave: boolean;
  onSave: () => void;
  onNameChange: (name: string) => void;
  onDescriptionChange: (description: string) => void;
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
  pinSavePopover,
}: SaveControlProps) => {
  /**
   * Note for improvement:
   * At the moment we are leveraging onDescriptionChange and onNameChange to handle the changes in the description and name of the agent.
   * We should migrate this to be handled with form controls and a form library.
   */

  const handleSave = useCallback(() => {
    onSave();
  }, [onSave]);

  const { toast } = useToast();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "s") {
        event.preventDefault(); // Stop the browser default action
        handleSave(); // Call your save function
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
  }, [handleSave, toast]);

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
      >
        <Card className="border-none shadow-none dark:bg-slate-900">
          <CardContent className="p-4">
            <div className="grid gap-3">
              <Label htmlFor="name" className="dark:text-gray-300">
                Name
              </Label>
              <Input
                id="name"
                placeholder="Enter your agent name"
                className="col-span-3"
                value={agentName}
                onChange={(e) => onNameChange(e.target.value)}
                data-id="save-control-name-input"
                data-testid="save-control-name-input"
                maxLength={100}
              />
              <Label htmlFor="description" className="dark:text-gray-300">
                Description
              </Label>
              <Input
                id="description"
                placeholder="Your agent description"
                className="col-span-3"
                value={agentDescription}
                onChange={(e) => onDescriptionChange(e.target.value)}
                data-id="save-control-description-input"
                data-testid="save-control-description-input"
                maxLength={500}
              />
              {agentMeta?.version && (
                <>
                  <Label htmlFor="version" className="dark:text-gray-300">
                    Version
                  </Label>
                  <Input
                    id="version"
                    placeholder="Version"
                    className="col-span-3"
                    value={agentMeta?.version || "-"}
                    disabled
                    data-testid="save-control-version-output"
                  />
                </>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex flex-col items-stretch gap-2">
            <Button
              className="w-full dark:bg-slate-700 dark:text-slate-100 dark:hover:bg-slate-800"
              onClick={handleSave}
              data-id="save-control-save-agent"
              data-testid="save-control-save-agent-button"
              disabled={!canSave}
            >
              Save Agent
            </Button>
          </CardFooter>
        </Card>
      </PopoverContent>
    </Popover>
  );
};
