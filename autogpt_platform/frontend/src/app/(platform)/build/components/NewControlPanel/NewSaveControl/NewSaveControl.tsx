import React from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Card, CardContent, CardFooter } from "@/components/__legacy__/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useNewSaveControl } from "./useNewSaveControl";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import { ControlPanelButton } from "../ControlPanelButton";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { FloppyDiskIcon } from "@phosphor-icons/react";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";

export const NewSaveControl = () => {
  const { form, onSubmit, isLoading, graphVersion } = useNewSaveControl({
    showToast: true,
  });
  const { saveControlOpen, setSaveControlOpen } = useControlPanelStore();
  return (
    <Popover onOpenChange={setSaveControlOpen}>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <ControlPanelButton
              data-id="save-control-popover-trigger"
              data-testid="save-control-save-button"
              selected={saveControlOpen}
              className="rounded-none"
            >
              <FloppyDiskIcon className="h-6 w-6" />
            </ControlPanelButton>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Save</TooltipContent>
      </Tooltip>
      <PopoverContent
        side="right"
        sideOffset={15}
        align="start"
        data-id="save-control-popover-content"
        className="w-96 max-w-[400px] rounded-xlarge"
      >
        <Card className="border-none dark:bg-slate-900">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)}>
              <CardContent className="p-0">
                <div className="space-y-3">
                  <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                      <Input
                        id="name"
                        label="Name"
                        size="small"
                        placeholder="Enter your agent name"
                        data-id="save-control-name-input"
                        data-testid="save-control-name-input"
                        maxLength={100}
                        wrapperClassName="!mb-0"
                        {...field}
                      />
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="description"
                    render={({ field }) => (
                      <Input
                        id="description"
                        size="small"
                        label="Description"
                        placeholder="Your agent description"
                        data-id="save-control-description-input"
                        data-testid="save-control-description-input"
                        maxLength={500}
                        wrapperClassName="!mb-0"
                        {...field}
                      />
                    )}
                  />

                  {graphVersion && (
                    <Input
                      id="version"
                      placeholder="Version"
                      size="small"
                      value={graphVersion || "-"}
                      disabled
                      data-testid="save-control-version-output"
                      label="Version"
                      wrapperClassName="!mb-0"
                    />
                  )}
                </div>
              </CardContent>
              {/* TODO: Add a cron schedule button */}
              <CardFooter className="mt-3 flex flex-col items-stretch gap-2 p-0">
                <Button
                  variant="primary"
                  type="submit"
                  size="small"
                  className="w-full dark:bg-slate-700 dark:text-slate-100 dark:hover:bg-slate-800"
                  data-id="save-control-save-agent"
                  data-testid="save-control-save-agent-button"
                  disabled={isLoading}
                  loading={isLoading}
                >
                  Save Agent
                </Button>
              </CardFooter>
            </form>
          </Form>
        </Card>
      </PopoverContent>
    </Popover>
  );
};
