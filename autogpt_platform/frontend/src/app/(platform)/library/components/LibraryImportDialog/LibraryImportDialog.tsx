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
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { UploadSimpleIcon } from "@phosphor-icons/react";
import { useLibraryImportWorkflowDialog } from "../LibraryImportWorkflowDialog/useLibraryImportWorkflowDialog";
import { useLibraryUploadAgentDialog } from "../LibraryUploadAgentDialog/useLibraryUploadAgentDialog";
import { useState } from "react";

// Only n8n template URLs are supported for direct URL fetching.
// Make.com and Zapier don't expose public JSON endpoints — use file upload instead.
const N8N_EXAMPLES = [
  { label: "Build Your First AI Agent", url: "https://n8n.io/workflows/6270" },
  { label: "Interactive AI Chat Agent", url: "https://n8n.io/workflows/5819" },
];

export default function LibraryImportDialog() {
  const [isOpen, setIsOpen] = useState(false);

  const importWorkflow = useLibraryImportWorkflowDialog();

  function handleClose() {
    setIsOpen(false);
    importWorkflow.setFileValue("");
    importWorkflow.setUrlValue("");
  }

  const upload = useLibraryUploadAgentDialog({ onSuccess: handleClose });

  return (
    <Dialog
      title="Import"
      styling={{ maxWidth: "32rem" }}
      controlled={{
        isOpen,
        set: setIsOpen,
      }}
      onClose={handleClose}
    >
      <Dialog.Trigger>
        <Button
          data-testid="import-button"
          variant="primary"
          className="h-[2.78rem] w-full md:w-[10rem]"
          size="small"
        >
          <UploadSimpleIcon width={18} height={18} />
          <span>Import</span>
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <TabsLine defaultValue="agent">
          <TabsLineList>
            <TabsLineTrigger value="agent">AutoGPT agent</TabsLineTrigger>
            <TabsLineTrigger value="platform">
              Import from another platform
            </TabsLineTrigger>
          </TabsLineList>

          {/* Tab: Import from any platform (file upload + n8n URL) */}
          <TabsLineContent value="platform">
            <p className="mb-4 text-sm text-neutral-500">
              Upload a workflow exported from n8n, Make.com, Zapier, or any
              other platform. AutoPilot will convert it into an AutoGPT agent
              for you.
            </p>
            <FileInput
              mode="base64"
              value={importWorkflow.fileValue}
              onChange={importWorkflow.setFileValue}
              accept=".json,application/json"
              placeholder="Workflow file (n8n, Make.com, Zapier, …)"
              maxFileSize={10 * 1024 * 1024}
              showStorageNote={false}
              className="mb-4 mt-2"
            />
            <Button
              type="button"
              variant="primary"
              className="w-full"
              disabled={!importWorkflow.fileValue}
              onClick={() => importWorkflow.submitWithMode("file")}
            >
              Import to AutoPilot
            </Button>

            <div className="my-5 flex items-center gap-3">
              <div className="h-px flex-1 bg-neutral-200" />
              <span className="text-xs text-neutral-400">
                or import from n8n marketplace
              </span>
              <div className="h-px flex-1 bg-neutral-200" />
            </div>

            <div className="mb-3 flex flex-wrap gap-2">
              {N8N_EXAMPLES.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => importWorkflow.setUrlValue(p.url)}
                  className="rounded-full border border-neutral-200 px-3 py-1 text-xs text-neutral-600 hover:border-purple-400 hover:text-purple-600"
                >
                  {p.label}
                </button>
              ))}
            </div>
            <Input
              id="template-url"
              value={importWorkflow.urlValue}
              onChange={(e) => importWorkflow.setUrlValue(e.target.value)}
              label="n8n workflow URL"
              placeholder="https://n8n.io/workflows/1234"
              className="mb-4 w-full rounded-[10px]"
            />
            <Button
              type="button"
              variant="primary"
              className="w-full"
              disabled={!importWorkflow.urlValue}
              onClick={() => importWorkflow.submitWithMode("url")}
            >
              Import from n8n
            </Button>
          </TabsLineContent>

          {/* Tab: Upload AutoGPT agent JSON */}
          <TabsLineContent value="agent">
            <p className="mb-4 text-sm text-neutral-500">
              Upload a previously exported AutoGPT agent file (.json).
            </p>
            <Form
              form={upload.form}
              onSubmit={upload.onSubmit}
              className="flex flex-col justify-center gap-0 px-1"
            >
              <FormField
                control={upload.form.control}
                name="agentName"
                render={({ field }) => (
                  <FormItem>
                    <FormControl>
                      <Input
                        {...field}
                        id={field.name}
                        label="Agent name"
                        className="w-full rounded-[10px]"
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={upload.form.control}
                name="agentDescription"
                render={({ field }) => (
                  <FormItem>
                    <FormControl>
                      <Input
                        {...field}
                        id={field.name}
                        label="Agent description"
                        type="textarea"
                        className="w-full rounded-[10px]"
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={upload.form.control}
                name="agentFile"
                render={({ field }) => (
                  <FormItem>
                    <FormControl>
                      <FileInput
                        mode="base64"
                        value={field.value}
                        onChange={field.onChange}
                        accept=".json,application/json"
                        placeholder="Agent file"
                        maxFileSize={10 * 1024 * 1024}
                        showStorageNote={false}
                        className="mb-8 mt-4"
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button
                type="submit"
                variant="primary"
                className="w-full"
                disabled={!upload.agentObject || upload.isUploading}
              >
                {upload.isUploading ? (
                  <div className="flex items-center gap-2">
                    <LoadingSpinner size="small" className="text-white" />
                    <span>Uploading...</span>
                  </div>
                ) : (
                  "Upload"
                )}
              </Button>
            </Form>
          </TabsLineContent>
        </TabsLine>
      </Dialog.Content>
    </Dialog>
  );
}
