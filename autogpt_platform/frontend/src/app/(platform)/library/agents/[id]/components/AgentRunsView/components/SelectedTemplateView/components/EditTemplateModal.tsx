"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { useState, useEffect } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePatchV2UpdateAnExistingPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { ModalHeader } from "../../RunAgentModal/components/ModalHeader/ModalHeader";
import { AgentCostSection } from "../../RunAgentModal/components/AgentCostSection/AgentCostSection";
import { AgentSectionHeader } from "../../RunAgentModal/components/AgentSectionHeader/AgentSectionHeader";
import { ModalRunSection } from "../../RunAgentModal/components/ModalRunSection/ModalRunSection";
import { RunAgentModalContextProvider } from "../../RunAgentModal/context";
import { AgentDetails } from "../../RunAgentModal/components/AgentDetails/AgentDetails";
import { Input } from "@/components/atoms/Input/Input";

interface Props {
  triggerSlot: React.ReactNode;
  agent: LibraryAgent;
  preset: LibraryAgentPreset;
  onSaved?: (updatedPreset: LibraryAgentPreset) => void;
}

export function EditTemplateModal({
  triggerSlot,
  agent,
  preset,
  onSaved,
}: Props) {
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [presetName, setPresetName] = useState(preset.name);
  const [presetDescription, setPresetDescription] = useState(
    preset.description,
  );
  const [inputValues, setInputValues] = useState<Record<string, any>>(
    preset.inputs || {},
  );
  const [inputCredentials, setInputCredentials] = useState<Record<string, any>>(
    preset.credentials || {},
  );

  // Reset form when preset changes
  useEffect(() => {
    setPresetName(preset.name);
    setPresetDescription(preset.description);
    setInputValues(preset.inputs || {});
    setInputCredentials(preset.credentials || {});
  }, [preset]);

  // Update preset mutation
  const updatePresetMutation = usePatchV2UpdateAnExistingPreset({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Template updated successfully",
            variant: "default",
          });
          setIsOpen(false);
          onSaved?.(response.data as unknown as LibraryAgentPreset);
        }
      },
      onError: (error: any) => {
        toast({
          title: "Failed to update template",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  // Input schema validation (reusing logic from useAgentRunModal)
  const agentInputSchema = agent.trigger_setup_info
    ? agent.trigger_setup_info.config_schema
    : agent.input_schema;
  const agentInputFields = (() => {
    if (
      !agentInputSchema ||
      typeof agentInputSchema !== "object" ||
      !("properties" in agentInputSchema) ||
      !agentInputSchema.properties
    ) {
      return {};
    }
    const properties = agentInputSchema.properties as Record<string, any>;
    return Object.fromEntries(
      Object.entries(properties).filter(
        ([_, subSchema]: [string, any]) => !subSchema.hidden,
      ),
    );
  })();

  const agentCredentialsInputFields = (() => {
    if (
      !agent.credentials_input_schema ||
      typeof agent.credentials_input_schema !== "object" ||
      !("properties" in agent.credentials_input_schema) ||
      !agent.credentials_input_schema.properties
    ) {
      return {} as Record<string, any>;
    }
    return agent.credentials_input_schema.properties as Record<string, any>;
  })();

  const hasAnySetupFields =
    Object.keys(agentInputFields || {}).length > 0 ||
    Object.keys(agentCredentialsInputFields || {}).length > 0;

  function handleInputChange(key: string, value: string) {
    setInputValues((prev) => ({
      ...prev,
      [key]: value,
    }));
  }

  function handleCredentialsChange(key: string, value: any | undefined) {
    setInputCredentials((prev) => {
      const next = { ...prev } as Record<string, any>;
      if (value === undefined) {
        delete next[key];
        return next;
      }
      next[key] = value;
      return next;
    });
  }

  function handleSetOpen(open: boolean) {
    setIsOpen(open);
  }

  function handleSave() {
    updatePresetMutation.mutate({
      presetId: preset.id,
      data: {
        name: presetName,
        description: presetDescription,
        inputs: inputValues,
        credentials: inputCredentials,
      },
    });
  }

  const hasChanges =
    presetName !== preset.name ||
    presetDescription !== preset.description ||
    JSON.stringify(inputValues) !== JSON.stringify(preset.inputs || {}) ||
    JSON.stringify(inputCredentials) !==
      JSON.stringify(preset.credentials || {});

  return (
    <Dialog
      controlled={{ isOpen, set: handleSetOpen }}
      styling={{ maxWidth: "600px", maxHeight: "90vh" }}
    >
      <Dialog.Trigger>{triggerSlot}</Dialog.Trigger>
      <Dialog.Content>
        <div className="flex h-full flex-col pb-4">
          {/* Header */}
          <div className="flex-shrink-0">
            <ModalHeader agent={agent} />
            <AgentCostSection flowId={agent.graph_id} />
          </div>

          {/* Scrollable content */}
          <div className="flex-1 pr-1" style={{ scrollbarGutter: "stable" }}>
            {/* Template Info Section */}
            <div className="mt-10">
              <AgentSectionHeader title="Template Information" />
              <div className="mb-10 mt-4 space-y-4">
                <div className="flex flex-col space-y-2">
                  <label className="text-sm font-medium">Template Name</label>
                  <Input
                    id="template_name"
                    label="Template Name"
                    size="small"
                    hideLabel
                    value={presetName}
                    placeholder="Enter template name"
                    onChange={(e) => setPresetName(e.target.value)}
                  />
                </div>
                <div className="flex flex-col space-y-2">
                  <label className="text-sm font-medium">
                    Template Description
                  </label>
                  <Input
                    id="template_description"
                    label="Template Description"
                    size="small"
                    hideLabel
                    value={presetDescription}
                    placeholder="Enter template description"
                    onChange={(e) => setPresetDescription(e.target.value)}
                  />
                </div>
              </div>
            </div>

            {/* Setup Section */}
            {hasAnySetupFields ? (
              <div className="mt-8">
                <RunAgentModalContextProvider
                  value={{
                    agent,
                    defaultRunType: "manual", // Always manual for templates
                    presetName,
                    setPresetName,
                    presetDescription,
                    setPresetDescription,
                    inputValues,
                    setInputValue: handleInputChange,
                    agentInputFields,
                    inputCredentials,
                    setInputCredentialsValue: handleCredentialsChange,
                    agentCredentialsInputFields,
                  }}
                >
                  <>
                    <AgentSectionHeader title="Template Setup" />
                    <ModalRunSection />
                  </>
                </RunAgentModalContextProvider>
              </div>
            ) : null}

            {/* Agent Details Section */}
            <div className="mt-8">
              <AgentSectionHeader title="Agent Details" />
              <AgentDetails agent={agent} />
            </div>
          </div>
        </div>
        <Dialog.Footer
          className="fixed bottom-1 left-0 z-10 w-full bg-white p-4"
          style={{ boxShadow: "0px -8px 10px white" }}
        >
          <div className="flex items-center justify-end gap-3">
            <Button
              variant="secondary"
              onClick={() => setIsOpen(false)}
              disabled={updatePresetMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleSave}
              disabled={
                !hasChanges ||
                updatePresetMutation.isPending ||
                !presetName.trim()
              }
            >
              {updatePresetMutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
