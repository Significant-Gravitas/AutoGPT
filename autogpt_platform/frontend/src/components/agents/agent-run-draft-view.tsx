"use client";
import React, { useCallback, useMemo, useState } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  CredentialsMetaInput,
  GraphExecutionID,
  LibraryAgent,
  LibraryAgentPreset,
} from "@/lib/autogpt-server-api";

import type { ButtonAction } from "@/components/agptui/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CredentialsInput } from "@/components/integrations/credentials-input";
import { TypeBasedInput } from "@/components/type-based-input";
import { useToastOnFail } from "@/components/ui/use-toast";
import ActionButtonGroup from "@/components/agptui/action-button-group";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import SchemaTooltip from "@/components/SchemaTooltip";
import { IconPlay } from "@/components/ui/icons";
import { useToast } from "@/components/ui/use-toast";
import { Input } from "@/components/ui/input";

export default function AgentRunDraftView({
  agent,
  onRun,
  onCreatePreset,
  agentActions,
}: {
  agent: LibraryAgent;
  onRun: (runID: GraphExecutionID) => void;
  onCreatePreset: (preset: LibraryAgentPreset) => void;
  agentActions: ButtonAction[];
}): React.ReactNode {
  const api = useBackendAPI();
  const { toast } = useToast();
  const toastOnFail = useToastOnFail();

  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const [presetName, setPresetName] = useState<string>("");
  const [presetDescription, setPresetDescription] = useState<string>("");
  const { state: onboardingState, completeStep: completeOnboardingStep } =
    useOnboarding();

  const agentInputs = useMemo(() => {
    if (agent.has_external_trigger) {
      return agent.trigger_setup_info.config_schema.properties;
    }
    return agent.input_schema.properties;
  }, [agent]);
  const agentCredentialsInputs = useMemo(
    () => agent.credentials_input_schema.properties,
    [agent],
  );

  const doRun = useCallback(() => {
    // Manually running webhook-triggered agents is not supported
    if (agent.has_external_trigger) return;

    api
      .executeGraph(
        agent.graph_id,
        agent.graph_version,
        inputValues,
        inputCredentials,
      )
      .then((newRun) => onRun(newRun.graph_exec_id))
      .catch(toastOnFail("execute agent"));
    // Mark run agent onboarding step as completed
    if (onboardingState?.completedSteps.includes("MARKETPLACE_ADD_AGENT")) {
      completeOnboardingStep("MARKETPLACE_RUN_AGENT");
    }
  }, [
    api,
    agent,
    inputValues,
    inputCredentials,
    onRun,
    toastOnFail,
    onboardingState,
    completeOnboardingStep,
  ]);

  const doSetupTrigger = useCallback(() => {
    // Setting up a trigger for non-webhook-triggered agents is not supported
    if (!agent.has_external_trigger) return;

    const credentialsInputName =
      agent.trigger_setup_info.credentials_input_name;

    if (!credentialsInputName) {
      // FIXME: implement support for manual-setup webhooks
      toast({
        variant: "destructive",
        title: "🚧 Feature under construction",
        description: "Setting up non-auto-setup triggers is not yet supported.",
      });
      return;
    }

    api
      .setupAgentTrigger(agent.id, {
        name: presetName || agent.name,
        description: presetDescription || agent.description,
        trigger_config: inputValues,
        agent_credentials: inputCredentials,
      })
      .then((newPreset) => onCreatePreset(newPreset))
      .catch(toastOnFail("set up agent trigger"));

    // Mark run agent onboarding step as completed(?)
    if (onboardingState?.completedSteps.includes("MARKETPLACE_ADD_AGENT")) {
      completeOnboardingStep("MARKETPLACE_RUN_AGENT");
    }
  }, [
    api,
    agent,
    presetName,
    presetDescription,
    inputValues,
    inputCredentials,
    onCreatePreset,
    toast,
    toastOnFail,
    onboardingState,
    completeOnboardingStep,
  ]);

  const runActions: ButtonAction[] = useMemo(
    () => [
      !agent.has_external_trigger
        ? {
            label: (
              <>
                <IconPlay className="mr-2 size-5" />
                Run
              </>
            ),
            variant: "accent",
            callback: doRun,
          }
        : {
            label: (
              <>
                <IconPlay className="mr-2 size-5" />
                Set up trigger
              </>
            ),
            variant: "accent",
            callback: doSetupTrigger,
          },
    ],
    [agent.has_external_trigger, doRun, doSetupTrigger],
  );

  return (
    <div className="agpt-div flex gap-6">
      <div className="flex flex-1 flex-col gap-4">
        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {agent.has_external_trigger && (
              <>
                {/* Preset name and description */}
                <div className="flex flex-col space-y-2">
                  <label className="flex items-center gap-1 text-sm font-medium">
                    Trigger Name
                    <SchemaTooltip description="Name of the trigger you are setting up" />
                  </label>
                  <Input
                    value={presetName}
                    placeholder="Enter trigger name"
                    onChange={(e) => setPresetName(e.target.value)}
                  />
                </div>
                <div className="flex flex-col space-y-2">
                  <label className="flex items-center gap-1 text-sm font-medium">
                    Trigger Description
                    <SchemaTooltip description="Description of the trigger you are setting up" />
                  </label>
                  <Input
                    value={presetDescription}
                    placeholder="Enter trigger description"
                    onChange={(e) => setPresetDescription(e.target.value)}
                  />
                </div>
              </>
            )}

            {/* Credentials inputs */}
            {Object.entries(agentCredentialsInputs).map(
              ([key, inputSubSchema]) => (
                <CredentialsInput
                  key={key}
                  schema={{ ...inputSubSchema, discriminator: undefined }}
                  selectedCredentials={
                    inputCredentials[key] ?? inputSubSchema.default
                  }
                  onSelectCredentials={(value) =>
                    setInputCredentials((obj) => {
                      const newObj = { ...obj };
                      if (value === undefined) {
                        delete newObj[key];
                        return newObj;
                      }
                      return {
                        ...obj,
                        [key]: value,
                      };
                    })
                  }
                  hideIfSingleCredentialAvailable={!agent.has_external_trigger}
                />
              ),
            )}

            {/* Regular inputs */}
            {Object.entries(agentInputs).map(([key, inputSubSchema]) => (
              <div key={key} className="flex flex-col space-y-2">
                <label className="flex items-center gap-1 text-sm font-medium">
                  {inputSubSchema.title || key}
                  <SchemaTooltip description={inputSubSchema.description} />
                </label>

                <TypeBasedInput
                  schema={inputSubSchema}
                  value={inputValues[key] ?? inputSubSchema.default}
                  placeholder={inputSubSchema.description}
                  onChange={(value) =>
                    setInputValues((obj) => ({
                      ...obj,
                      [key]: value,
                    }))
                  }
                />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Actions */}
      <aside className="w-48 xl:w-56">
        <div className="flex flex-col gap-8">
          <ActionButtonGroup title="Run actions" actions={runActions} />

          <ActionButtonGroup title="Agent actions" actions={agentActions} />
        </div>
      </aside>
    </div>
  );
}
