"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  GraphExecutionID,
  GraphMeta,
  LibraryAgentPreset,
} from "@/lib/autogpt-server-api";

import type { ButtonAction } from "@/components/agptui/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TypeBasedInput } from "@/components/type-based-input";
import { useToastOnFail } from "@/components/ui/use-toast";
import ActionButtonGroup from "@/components/agptui/action-button-group";
import SchemaTooltip from "@/components/SchemaTooltip";
import { IconPlay } from "@/components/ui/icons";
import { deepEquals } from "@/lib/utils";

export default function AgentRunDraftView({
  graph,
  preset,
  onRun,
  onSavePreset,
  agentActions,
}: {
  graph: GraphMeta;
  preset?: LibraryAgentPreset;
  onRun: (runID: GraphExecutionID) => void;
  onSavePreset: (preset: LibraryAgentPreset) => void;
  agentActions: ButtonAction[];
}): React.ReactNode {
  const api = useBackendAPI();
  const toastOnFail = useToastOnFail();

  const agentInputs = graph.input_schema.properties;
  const [inputValues, setInputValues] = useState<Record<string, any>>({});

  useEffect(() => {
    if (preset == undefined) {
      setInputValues({});
      return;
    }
    if (deepEquals(preset.inputs, inputValues)) return;
    setInputValues(preset.inputs);
    // Sets `inputValues` once if `preset` is changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset]);

  const doRun = useCallback(
    () =>
      api
        .executeGraph(graph.id, graph.version, inputValues)
        .then((newRun) => onRun(newRun.graph_exec_id))
        .catch(toastOnFail("execute agent")),
    [api, graph, inputValues, onRun, toastOnFail],
  );

  const savePreset = useCallback(
    () =>
      (preset
        ? // Update existing preset
          api.updateLibraryAgentPreset(preset.id, {
            ...preset, // TODO: update specific attributes
            inputs: inputValues,
          })
        : // Save run draft as new preset
          api.createLibraryAgentPreset({
            graph_id: graph.id,
            graph_version: graph.version,
            name: graph.name,
            description: "", // TODO: add dialog for name + description
            inputs: inputValues,
            is_active: true,
          })
      ).then(onSavePreset),
    [
      preset,
      api,
      inputValues,
      graph.id,
      graph.version,
      graph.name,
      onSavePreset,
    ],
  );

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
      {
        label: preset ? "Save preset" : "Save as a preset",
        callback: savePreset,
      },
    ],
    [doRun, preset, savePreset],
  );

  return (
    <div className="agpt-div flex gap-6">
      <div className="flex flex-1 flex-col gap-4">
        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
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
