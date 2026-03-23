"use client";
import { Button } from "@/components/atoms/Button/Button";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { Input } from "@/components/atoms/Input/Input";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
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
import AgentUploadTabContent from "./AgentUploadTabContent";

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
            <TabsLineTrigger value="platform">Another platform</TabsLineTrigger>
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
              disabled={
                !importWorkflow.fileValue || importWorkflow.isSubmitting
              }
              onClick={() => importWorkflow.submitWithMode("file")}
            >
              {importWorkflow.submittingMode === "file" ? (
                <div className="flex items-center gap-2">
                  <LoadingSpinner size="small" className="text-white" />
                  <span>Importing...</span>
                </div>
              ) : (
                "Import to AutoPilot"
              )}
            </Button>

            <div className="my-5 flex items-center gap-3">
              <div className="h-px flex-1 bg-neutral-200" />
              <span className="text-xs text-neutral-400">
                or import from URL
              </span>
              <div className="h-px flex-1 bg-neutral-200" />
            </div>

            <div className="mb-3 flex flex-wrap gap-2">
              {N8N_EXAMPLES.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  disabled={importWorkflow.isSubmitting}
                  onClick={() => importWorkflow.setUrlValue(p.url)}
                  className="rounded-full border border-neutral-200 px-3 py-1 text-xs text-neutral-600 hover:border-purple-400 hover:text-purple-600 disabled:opacity-50"
                >
                  {p.label}
                </button>
              ))}
            </div>
            <Input
              id="template-url"
              value={importWorkflow.urlValue}
              onChange={(e) => importWorkflow.setUrlValue(e.target.value)}
              label="Workflow URL"
              placeholder="https://n8n.io/workflows/1234"
              className="mb-4 w-full rounded-[10px]"
            />
            <Button
              type="button"
              variant="primary"
              className="w-full"
              disabled={!importWorkflow.urlValue || importWorkflow.isSubmitting}
              onClick={() => importWorkflow.submitWithMode("url")}
            >
              {importWorkflow.submittingMode === "url" ? (
                <div className="flex items-center gap-2">
                  <LoadingSpinner size="small" className="text-white" />
                  <span>Importing...</span>
                </div>
              ) : (
                "Import from URL"
              )}
            </Button>
          </TabsLineContent>

          {/* Tab: Upload AutoGPT agent JSON */}
          <AgentUploadTabContent upload={upload} />
        </TabsLine>
      </Dialog.Content>
    </Dialog>
  );
}
