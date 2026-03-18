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
import { useRouter } from "next/navigation";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";

const PLATFORM_EXAMPLES = [
  {
    name: "n8n",
    url: "https://n8n.io/workflows/1234-example-workflow",
  },
  {
    name: "Make.com",
    url: "https://www.make.com/en/templates/12345",
  },
  {
    name: "Zapier",
    url: "https://zapier.com/shared/abc123",
  },
];

export default function LibraryImportDialog() {
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();
  const router = useRouter();

  const upload = useLibraryUploadAgentDialog();
  const importWorkflow = useLibraryImportWorkflowDialog();

  function handleClose() {
    setIsOpen(false);
    importWorkflow.setFileValue("");
    importWorkflow.setUrlValue("");
  }

  function submitImport(mode: "url" | "file") {
    let prompt: string;
    if (mode === "url" && importWorkflow.urlValue) {
      prompt = `Import this workflow and recreate it as an AutoGPT agent: ${importWorkflow.urlValue}`;
    } else if (mode === "file" && importWorkflow.fileValue) {
      const base64Match = importWorkflow.fileValue.match(
        /^data:[^;]+;base64,(.+)$/,
      );
      if (!base64Match) {
        toast({
          title: "Invalid file",
          description: "Could not read the uploaded file.",
          variant: "destructive",
        });
        return;
      }
      try {
        const jsonString = atob(base64Match[1]);
        JSON.parse(jsonString);
        prompt = `Import this workflow JSON and recreate it as an AutoGPT agent:\n\`\`\`json\n${jsonString}\n\`\`\``;
      } catch {
        toast({
          title: "Invalid JSON",
          description: "The uploaded file is not valid JSON.",
          variant: "destructive",
        });
        return;
      }
    } else {
      return;
    }

    handleClose();
    toast({
      title: "Redirecting to AutoPilot",
      description: "AutoPilot will import and convert the workflow for you.",
    });
    sessionStorage.setItem("importWorkflowPrompt", prompt);
    router.push("/copilot?source=import&autosubmit=true");
  }

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
            <TabsLineTrigger value="agent">Upload agent</TabsLineTrigger>
            <TabsLineTrigger value="url">From URL</TabsLineTrigger>
            <TabsLineTrigger value="workflow-file">
              Upload workflow
            </TabsLineTrigger>
          </TabsLineList>

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
                className="min-w-[18rem]"
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

          {/* Tab: Import from competitor URL */}
          <TabsLineContent value="url">
            <p className="mb-3 text-sm text-neutral-500">
              Paste a workflow URL from n8n, Make.com, or Zapier. AutoPilot will
              automatically convert it to an AutoGPT agent.
            </p>
            <div className="mb-4 flex flex-wrap gap-2">
              {PLATFORM_EXAMPLES.map((p) => (
                <button
                  key={p.name}
                  type="button"
                  onClick={() => importWorkflow.setUrlValue(p.url)}
                  className="rounded-full border border-neutral-200 px-3 py-1 text-xs text-neutral-600 hover:border-purple-400 hover:text-purple-600"
                >
                  {p.name} example
                </button>
              ))}
            </div>
            <Input
              id="template-url"
              value={importWorkflow.urlValue}
              onChange={(e) => importWorkflow.setUrlValue(e.target.value)}
              label="Workflow URL"
              placeholder="https://n8n.io/workflows/1234  •  make.com/templates/…  •  zapier.com/shared/…"
              className="mb-4 w-full rounded-[10px]"
            />
            <Button
              type="button"
              variant="primary"
              className="min-w-[18rem]"
              disabled={!importWorkflow.urlValue}
              onClick={() => submitImport("url")}
            >
              Import to AutoPilot
            </Button>
          </TabsLineContent>

          {/* Tab: Upload competitor workflow file */}
          <TabsLineContent value="workflow-file">
            <p className="mb-4 text-sm text-neutral-500">
              Upload a workflow file exported from n8n, Make.com, or Zapier.
              AutoPilot will automatically convert it to an AutoGPT agent.
            </p>
            <FileInput
              mode="base64"
              value={importWorkflow.fileValue}
              onChange={importWorkflow.setFileValue}
              accept=".json,application/json"
              placeholder="Workflow JSON file (n8n, Make.com, or Zapier export)"
              maxFileSize={10 * 1024 * 1024}
              showStorageNote={false}
              className="mb-4 mt-2"
            />
            <Button
              type="button"
              variant="primary"
              className="min-w-[18rem]"
              disabled={!importWorkflow.fileValue}
              onClick={() => submitImport("file")}
            >
              Import to AutoPilot
            </Button>
          </TabsLineContent>
        </TabsLine>
      </Dialog.Content>
    </Dialog>
  );
}
