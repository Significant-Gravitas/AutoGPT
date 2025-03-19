"use client";
import React, { useCallback, useMemo, useState } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { GraphExecutionID, GraphMeta } from "@/lib/autogpt-server-api";

import type { ButtonAction } from "@/components/agptui/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LocalValuedInput } from "@/components/ui/input";
import { Pencil2Icon } from "@radix-ui/react-icons";
import { Textarea } from "@/components/ui/textarea";
import { IconPlay } from "@/components/ui/icons";
import { Button } from "@/components/agptui/Button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

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

  const agentInputs = graph.input_schema.properties;
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [expandedInputKey, setExpandedInputKey] = useState<string | null>(null);
  const [tempInputValue, setTempInputValue] = useState("");

  const openInputPopout = useCallback(
    (key: string) => {
      setTempInputValue(inputValues[key] || "");
      setExpandedInputKey(key);
    },
    [inputValues],
  );

  const closeInputPopout = useCallback(() => {
    setExpandedInputKey(null);
  }, []);

  const saveAndCloseInputPopout = useCallback(() => {
    if (!expandedInputKey) return;

    setInputValues((obj) => ({ ...obj, [expandedInputKey]: tempInputValue }));
    closeInputPopout();
  }, [expandedInputKey, tempInputValue, closeInputPopout]);

  const doRun = useCallback(
    () =>
      api
        .executeGraph(graph.id, graph.version, inputValues)
        .then((newRun) => onRun(newRun.graph_exec_id)),
    [api, graph, inputValues, onRun],
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
            {Object.entries(agentInputs).map(([key, inputSubSchema]) => (
              <div key={key} className="flex flex-col gap-1.5">
                <label className="text-sm font-medium">
                  {inputSubSchema.title || key}
                </label>

                <div className="nodrag relative">
                  <LocalValuedInput
                    // TODO: render specific inputs based on input types
                    defaultValue={
                      "default" in inputSubSchema ? inputSubSchema.default : ""
                    }
                    value={inputValues[key] ?? undefined}
                    className="rounded-full pr-8"
                    onChange={(e) =>
                      setInputValues((obj) => ({
                        ...obj,
                        [key]: e.target.value,
                      }))
                    }
                  />

                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute inset-1 left-auto h-7 w-8 rounded-full"
                    onClick={() => openInputPopout(key)}
                    title="Open a larger textbox input"
                  >
                    <Pencil2Icon className="m-0 p-0" />
                  </Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Pop-out Long Input Modal */}
      <Dialog
        open={expandedInputKey !== null}
        onOpenChange={(open) => !open && closeInputPopout()}
      >
        <DialogContent className="sm:max-w-[720px]">
          <DialogHeader>
            <DialogTitle>
              Edit {expandedInputKey && agentInputs[expandedInputKey].title}
            </DialogTitle>
          </DialogHeader>
          <Textarea
            value={tempInputValue}
            onChange={(e) => setTempInputValue(e.target.value)}
            className="min-h-[320px]"
          />
          <div className="flex justify-end gap-2">
            <Button onClick={closeInputPopout}>Cancel</Button>
            <Button variant="primary" onClick={saveAndCloseInputPopout}>
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Actions */}
      <aside className="w-48 xl:w-56">
        <div className="flex flex-col gap-8">
          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-medium">Run actions</h3>
            {runActions.map((action, i) => (
              <Button
                key={i}
                variant={action.variant ?? "outline"}
                onClick={action.callback}
              >
                {action.label}
              </Button>
            ))}
          </div>

          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-medium">Agent actions</h3>
            {agentActions.map((action, i) => (
              <Button
                key={i}
                variant={action.variant ?? "outline"}
                onClick={action.callback}
              >
                {action.label}
              </Button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
