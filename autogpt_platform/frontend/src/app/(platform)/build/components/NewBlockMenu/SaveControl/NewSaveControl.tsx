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
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ControlPanelButton } from "../ControlPanelButton";

interface SaveControlProps {
  agentMeta: GraphMeta | null;
  agentName: string;
  agentDescription: string;
  canSave: boolean;
  onSave: () => void;
  onNameChange: (name: string) => void;
  onDescriptionChange: (description: string) => void;
  pinSavePopover: boolean;

  blockMenuSelected: "save" | "block" | "search" | "";
  setBlockMenuSelected: React.Dispatch<
    React.SetStateAction<"" | "save" | "block" | "search">
  >;
}

export const NewSaveControl = ({
  agentMeta,
  canSave,
  onSave,
  agentName,
  onNameChange,
  agentDescription,
  onDescriptionChange,
  blockMenuSelected,
  setBlockMenuSelected,
  pinSavePopover,
}: SaveControlProps) => {

  const handleSave = useCallback(() => {
    onSave();
  }, [onSave]);

  const { toast } = useToast();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "s") {
        event.preventDefault(); 
        handleSave(); 
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
    <Popover
      open={pinSavePopover ? true : undefined}
      onOpenChange={(open) => open || setBlockMenuSelected("")}
    >
      <PopoverTrigger>
        <ControlPanelButton
          data-id="save-control-popover-trigger"
          data-testid="blocks-control-save-button"
          selected={blockMenuSelected === "save"}
          onClick={() => {
            setBlockMenuSelected("save");
          }}
          className="rounded-none"
        >
          {/* Need to find phosphor icon alternative for this lucide icon */}
          <IconSave className="h-5 w-5" strokeWidth={2} />
        </ControlPanelButton>
      </PopoverTrigger>

      <PopoverContent
        side="right"
        sideOffset={16}
        align="start"
        className="w-[17rem] rounded-xl border-none p-0 shadow-none md:w-[30rem]"
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