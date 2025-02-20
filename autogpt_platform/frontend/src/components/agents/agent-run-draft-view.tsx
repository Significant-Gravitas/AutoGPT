"use client";
import React, { useCallback, useMemo, useState } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { GraphMeta } from "@/lib/autogpt-server-api";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button, ButtonProps } from "@/components/agptui/Button";
import { Input } from "@/components/ui/input";

export default function AgentRunDraftView({
  agent,
  onRun,
  agentActions,
}: {
  agent: GraphMeta;
  onRun: (runID: string) => void;
  agentActions: { label: string; callback: () => void }[];
}): React.ReactNode {
  const api = useBackendAPI();

  const agentInputs = agent.input_schema.properties;
  const [inputValues, setInputValues] = useState<Record<string, any>>({});

  const doRun = useCallback(
    () =>
      api
        .executeGraph(agent.id, agent.version, inputValues)
        .then((newRun) => onRun(newRun.graph_exec_id)),
    [api, agent, inputValues, onRun],
  );

  const runActions: {
    label: string;
    variant?: ButtonProps["variant"];
    callback: () => void;
  }[] = useMemo(
    () => [{ label: "Run", variant: "accent", callback: () => doRun() }],
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
                <Input
                  // TODO: render specific inputs based on input types
                  defaultValue={
                    "default" in inputSubSchema ? inputSubSchema.default : ""
                  }
                  className="rounded-full"
                  onChange={(e) =>
                    setInputValues((obj) => ({ ...obj, [key]: e.target.value }))
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
              <Button key={i} variant="outline" onClick={action.callback}>
                {action.label}
              </Button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
