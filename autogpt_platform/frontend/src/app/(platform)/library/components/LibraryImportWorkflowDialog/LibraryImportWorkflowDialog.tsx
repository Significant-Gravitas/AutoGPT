"use client";
import { Button } from "@/components/atoms/Button/Button";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { Input } from "@/components/atoms/Input/Input";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/molecules/Form/Form";
import { ArrowsClockwiseIcon } from "@phosphor-icons/react";
import { z } from "zod";
import { useLibraryImportWorkflowDialog } from "./useLibraryImportWorkflowDialog";

export const importWorkflowFormSchema = z.object({
  workflowFile: z.string(),
  templateUrl: z.string(),
});

export default function LibraryImportWorkflowDialog() {
  const {
    onSubmit,
    isConverting,
    isOpen,
    setIsOpen,
    form,
    importMode,
    setImportMode,
  } = useLibraryImportWorkflowDialog();

  const hasInput =
    importMode === "url"
      ? !!form.watch("templateUrl")
      : !!form.watch("workflowFile");

  return (
    <Dialog
      title="Import Workflow"
      styling={{ maxWidth: "32rem" }}
      controlled={{
        isOpen,
        set: setIsOpen,
      }}
      onClose={() => {
        setIsOpen(false);
        form.reset();
      }}
    >
      <Dialog.Trigger>
        <Button
          data-testid="import-workflow-button"
          variant="primary"
          className="h-[2.78rem] w-full md:w-[14rem]"
          size="small"
        >
          <ArrowsClockwiseIcon width={18} height={18} />
          <span>Import workflow</span>
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        {/* Mode toggle */}
        <div className="mb-4 flex gap-2">
          <Button
            variant={importMode === "file" ? "primary" : "outline"}
            size="small"
            onClick={() => setImportMode("file")}
            type="button"
          >
            Upload file
          </Button>
          <Button
            variant={importMode === "url" ? "primary" : "outline"}
            size="small"
            onClick={() => setImportMode("url")}
            type="button"
          >
            Paste URL
          </Button>
        </div>

        <p className="mb-4 text-sm text-neutral-500">
          Import workflows from n8n, Make.com, or Zapier. The workflow will be
          automatically converted to an AutoGPT agent.
        </p>

        <Form
          form={form}
          onSubmit={onSubmit}
          className="flex flex-col justify-center gap-0 px-1"
        >
          {importMode === "file" ? (
            <FormField
              control={form.control}
              name="workflowFile"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <FileInput
                      mode="base64"
                      value={field.value}
                      onChange={field.onChange}
                      accept=".json,application/json"
                      placeholder="Workflow JSON file (n8n, Make.com, or Zapier export)"
                      maxFileSize={10 * 1024 * 1024}
                      showStorageNote={false}
                      className="mb-4 mt-2"
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          ) : (
            <FormField
              control={form.control}
              name="templateUrl"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <Input
                      {...field}
                      id={field.name}
                      label="n8n template URL"
                      placeholder="https://n8n.io/workflows/1234"
                      className="mb-4 mt-2 w-full rounded-[10px]"
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          )}

          <Button
            type="submit"
            variant="primary"
            className="min-w-[18rem]"
            disabled={!hasInput || isConverting}
          >
            {isConverting ? (
              <div className="flex items-center gap-2">
                <LoadingSpinner size="small" className="text-white" />
                <span>Parsing workflow...</span>
              </div>
            ) : (
              "Import to CoPilot"
            )}
          </Button>
        </Form>
      </Dialog.Content>
    </Dialog>
  );
}
