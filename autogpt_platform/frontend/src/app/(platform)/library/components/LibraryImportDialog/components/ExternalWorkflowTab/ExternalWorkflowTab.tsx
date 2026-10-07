"use client";
import { Button } from "@/components/atoms/Button/Button";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { Input } from "@/components/atoms/Input/Input";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { TabsLineContent } from "@/components/molecules/TabsLine/TabsLine";
import { useExternalWorkflowTab } from "./useExternalWorkflowTab";

const N8N_EXAMPLES = [
  { label: "Build Your First AI Agent", url: "https://n8n.io/workflows/6270" },
  { label: "Interactive AI Chat Agent", url: "https://n8n.io/workflows/5819" },
];

type ExternalWorkflowTabProps = {
  importWorkflow: ReturnType<typeof useExternalWorkflowTab>;
};

export default function ExternalWorkflowTab({
  importWorkflow,
}: ExternalWorkflowTabProps) {
  return (
    <TabsLineContent value="platform">
      <Text variant="body" tone="muted" className="mb-4">
        Upload a workflow exported from n8n, Make.com, Zapier, or any other
        platform. Otto will convert it into an AutoGPT agent for you.
      </Text>
      <FileInput
        mode="base64"
        value={importWorkflow.fileValue}
        onChange={importWorkflow.setFileValue}
        accept=".json,application/json"
        placeholder="Workflow file (n8n, Make.com, Zapier, ...)"
        maxFileSize={10 * 1024 * 1024}
        showStorageNote={false}
        className="mt-2 mb-4"
      />
      <Button
        type="button"
        variant="primary"
        className="w-full"
        disabled={!importWorkflow.fileValue || importWorkflow.isSubmitting}
        onClick={() => importWorkflow.submitWithMode("file")}
      >
        {importWorkflow.submittingMode === "file" ? (
          <div className="flex items-center gap-2">
            <LoadingSpinner size="small" className="text-white" />
            <span>Importing...</span>
          </div>
        ) : (
          "Import to Otto"
        )}
      </Button>

      <div className="my-5 flex items-center gap-3">
        <div className="h-px flex-1 bg-zinc-200" />
        <Text variant="small" as="span" tone="muted">
          or import from URL
        </Text>
        <div className="h-px flex-1 bg-zinc-200" />
      </div>

      <div className="mb-3 flex flex-wrap gap-2">
        {N8N_EXAMPLES.map((p) => (
          <Button
            key={p.label}
            type="button"
            variant="secondary"
            size="sm"
            disabled={importWorkflow.isSubmitting}
            onClick={() => importWorkflow.setUrlValue(p.url)}
            className="rounded-full font-normal text-zinc-600 shadow-none hover:border-purple-400 hover:bg-white hover:text-purple-600"
          >
            {p.label}
          </Button>
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
  );
}
