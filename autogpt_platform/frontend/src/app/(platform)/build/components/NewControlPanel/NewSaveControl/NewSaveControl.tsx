import { FormProvider } from "react-hook-form";
import { Card, CardContent, CardFooter } from "@/components/atoms/Card/Card";
import { FormField } from "@/components/molecules/Form/Form";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { ControlPanelButton } from "../ControlPanelButton";
import { useNewSaveControl } from "./useNewSaveControl";
import { FloppyDiskIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export const NewSaveControl = () => {
  const { form, isSaving, graphVersion, handleSave } = useNewSaveControl();
  const { saveControlOpen, setSaveControlOpen, forceOpenSave } =
    useControlPanelStore();

  return (
    <Popover
      onOpenChange={(open) => {
        if (!forceOpenSave || open) {
          setSaveControlOpen(open);
        }
      }}
      open={forceOpenSave ? true : saveControlOpen}
    >
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <ControlPanelButton
              data-id="save-control-popover-trigger"
              data-testid="save-control-save-button"
              selected={saveControlOpen}
              className="rounded-none"
            >
              <Icon icon={FloppyDiskIcon} className="size-5" />
            </ControlPanelButton>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Save</TooltipContent>
      </Tooltip>
      {/* TODO(tokens): today's popover look. No exact role: border-neutral-200
          #E5E5E5; text-neutral-950 #0A0A0A is the `focus` role, a ring colour,
          not a text colour (SECRT-3154). */}
      <PopoverContent
        side="right"
        sideOffset={15}
        align="start"
        data-id="save-control-popover-content"
        className="w-96 max-w-[400px] rounded-xlarge border-neutral-200 text-neutral-950 dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-50"
      >
        <Card
          variant="outlined"
          // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:bg-slate-900 #2A313A
          className="border-none dark:bg-slate-900"
        >
          <FormProvider {...form}>
            <form onSubmit={form.handleSubmit(handleSave)}>
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
                        placeholder="Enter your workflow name"
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
                        placeholder="Your workflow description"
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
                      data-tutorial-id="save-control-version-output"
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
                  // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:bg-slate-700 #475263, dark:text-slate-100 #EFF1F4, dark:hover:bg-slate-800 #475263
                  className="w-full dark:bg-slate-700 dark:text-slate-100 dark:hover:bg-slate-800"
                  data-id="save-control-save-agent"
                  data-testid="save-control-save-agent-button"
                  disabled={isSaving}
                  loading={isSaving}
                >
                  Save Workflow
                </Button>
              </CardFooter>
            </form>
          </FormProvider>
        </Card>
      </PopoverContent>
    </Popover>
  );
};
