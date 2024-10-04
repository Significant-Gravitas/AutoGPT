import React from "react";
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

interface SaveControlProps {
  agentMeta: GraphMeta | null;
  agentName: string;
  agentDescription: string;
  onSave: (isTemplate: boolean | undefined) => void;
  onNameChange: (name: string) => void;
  onDescriptionChange: (description: string) => void;
  pinSavePopover: boolean;
}

/**
 * A SaveControl component to be used within the ControlPanel. It allows the user to save the agent / template.
 * @param {Object} SaveControlProps - The properties of the SaveControl component.
 * @param {GraphMeta | null} SaveControlProps.agentMeta - The agent's metadata, or null if creating a new agent.
 * @param {(isTemplate: boolean | undefined) => void} SaveControlProps.onSave - Function to save the agent or template.
 * @param {(name: string) => void} SaveControlProps.onNameChange - Function to handle name changes.
 * @param {(description: string) => void} SaveControlProps.onDescriptionChange - Function to handle description changes.
 * @returns The SaveControl component.
 */
export const SaveControl = ({
  agentMeta,
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

  // Determines if we're saving a template or an agent
  let isTemplate = agentMeta?.is_template ? true : undefined;
  const handleSave = () => {
    onSave(isTemplate);
  };

  const getType = () => {
    return agentMeta?.is_template ? "template" : "agent";
  };

  return (
    <Popover open={pinSavePopover ? true : undefined}>
      <Tooltip delayDuration={500}>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              data-id="save-control-popover-trigger"
            >
              <IconSave />
            </Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Save</TooltipContent>
      </Tooltip>
      <PopoverContent side="right" sideOffset={15} align="start">
        <Card className="border-none shadow-none">
          <CardContent className="p-4">
            <div className="grid gap-3">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                placeholder="Enter your agent name"
                className="col-span-3"
                value={agentName}
                onChange={(e) => onNameChange(e.target.value)}
                data-id="save-control-name-input"
              />
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                placeholder="Your agent description"
                className="col-span-3"
                value={agentDescription}
                onChange={(e) => onDescriptionChange(e.target.value)}
                data-id="save-control-description-input"
              />
              {agentMeta?.version && (
                <>
                  <Label htmlFor="version">Version</Label>
                  <Input
                    id="version"
                    placeholder="Version"
                    className="col-span-3"
                    value={agentMeta?.version || "-"}
                    disabled
                  />
                </>
              )}
            </div>
          </CardContent>
          <CardFooter className="flex flex-col items-stretch gap-2">
            <Button
              className="w-full"
              onClick={handleSave}
              data-id="save-control-save-agent"
            >
              Save {getType()}
            </Button>
            {!agentMeta && (
              <Button
                variant="secondary"
                className="w-full"
                data-id="save-control-template-button"
                onClick={() => {
                  isTemplate = true;
                  handleSave();
                }}
              >
                Save as Template
              </Button>
            )}
          </CardFooter>
        </Card>
      </PopoverContent>
    </Popover>
  );
};
