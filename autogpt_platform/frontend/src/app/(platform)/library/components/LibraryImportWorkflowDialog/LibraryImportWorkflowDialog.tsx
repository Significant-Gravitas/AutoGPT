"use client";
import { Button } from "@/components/atoms/Button/Button";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ArrowsClockwiseIcon } from "@phosphor-icons/react";
import { useLibraryImportWorkflowDialog } from "./useLibraryImportWorkflowDialog";

export default function LibraryImportWorkflowDialog() {
  const {
    onSubmit,
    isOpen,
    setIsOpen,
    importMode,
    setImportMode,
    hasInput,
    fileValue,
    setFileValue,
    urlValue,
    setUrlValue,
  } = useLibraryImportWorkflowDialog();

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
        setFileValue("");
        setUrlValue("");
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
          Import workflows from n8n, Make.com, or Zapier. AutoPilot will
          automatically convert it to an AutoGPT agent.
        </p>

        {importMode === "file" ? (
          <FileInput
            mode="base64"
            value={fileValue}
            onChange={setFileValue}
            accept=".json,application/json"
            placeholder="Workflow JSON file (n8n, Make.com, or Zapier export)"
            maxFileSize={10 * 1024 * 1024}
            showStorageNote={false}
            className="mb-4 mt-2"
          />
        ) : (
          <Input
            id="template-url"
            value={urlValue}
            onChange={(e) => setUrlValue(e.target.value)}
            label="n8n template URL"
            placeholder="https://n8n.io/workflows/1234"
            className="mb-4 mt-2 w-full rounded-[10px]"
          />
        )}

        <Button
          type="button"
          variant="primary"
          className="min-w-[18rem]"
          disabled={!hasInput}
          onClick={onSubmit}
        >
          Import to AutoPilot
        </Button>
      </Dialog.Content>
    </Dialog>
  );
}
