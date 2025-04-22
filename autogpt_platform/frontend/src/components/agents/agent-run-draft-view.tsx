"use client";
import React, { useCallback, useMemo, useState } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { GraphExecutionID, GraphMeta } from "@/lib/autogpt-server-api";

import type { ButtonAction } from "@/components/agptui/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CredentialsInput } from "@/components/integrations/credentials-input";
import { TypeBasedInput } from "@/components/type-based-input";
import { useToastOnFail } from "@/components/ui/use-toast";
import ActionButtonGroup from "@/components/agptui/action-button-group";
import SchemaTooltip from "@/components/SchemaTooltip";
import { IconPlay } from "@/components/ui/icons";
import { useOnboarding } from "../onboarding/onboarding-provider";

export default function AgentRunDraftView({
  graph,
  onRun,
  agentActions,
}: {
  graph: GraphMeta;
  onRun: (runID: GraphExecutionID) => void;
  agentActions: ButtonAction[];
}): React.ReactNode {
  const api = useBackendAPI();
  const toastOnFail = useToastOnFail();

  const agentInputs = graph.input_schema.properties;
  const agentCredentialsInputs = graph.credentials_input_schema.properties;
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [inputCredentials, setInputCredentials] = useState<Record<string, any>>(
    {},
  );
  const { state, completeStep } = useOnboarding();

  const doRun = useCallback(() => {
    api
      .executeGraph(graph.id, graph.version, inputValues, inputCredentials)
      .then((newRun) => onRun(newRun.graph_exec_id))
      .catch(toastOnFail("execute agent"));
    // Mark run agent onboarding step as completed
    if (state?.completedSteps.includes("MARKETPLACE_ADD_AGENT")) {
      completeStep("MARKETPLACE_RUN_AGENT");
    }
  }, [
    api,
    graph,
    inputValues,
    inputCredentials,
    onRun,
    toastOnFail,
    state,
    completeStep,
  ]);

  const runActions: ButtonAction[] = useMemo(
    () => [
      {
        label: (
          <>
            <IconPlay className="mr-2 size-5" />
            Run
          </>
        ),
        variant: "accent",
        callback: doRun,
      },
    ],
    [doRun],
  );

  return (
    <div className="agpt-div flex gap-6">
      <div className="flex flex-1 flex-col gap-4">
        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
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
                    setInputCredentials((obj) => ({
                      ...obj,
                      [key]: value,
                    }))
                  }
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
