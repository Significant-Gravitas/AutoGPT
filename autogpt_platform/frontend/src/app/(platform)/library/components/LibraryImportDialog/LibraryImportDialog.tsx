"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { UploadSimpleIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { useLibraryUploadAgentDialog } from "../LibraryUploadAgentDialog/useLibraryUploadAgentDialog";
import AgentUploadTab from "./components/AgentUploadTab/AgentUploadTab";
import ExternalWorkflowTab from "./components/ExternalWorkflowTab/ExternalWorkflowTab";
import { useExternalWorkflowTab } from "./components/ExternalWorkflowTab/useExternalWorkflowTab";

export default function LibraryImportDialog() {
  const [isOpen, setIsOpen] = useState(false);

  const importWorkflow = useExternalWorkflowTab();

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
          <ExternalWorkflowTab importWorkflow={importWorkflow} />

          {/* Tab: Upload AutoGPT agent JSON */}
          <AgentUploadTab upload={upload} />
        </TabsLine>
      </Dialog.Content>
    </Dialog>
  );
}
