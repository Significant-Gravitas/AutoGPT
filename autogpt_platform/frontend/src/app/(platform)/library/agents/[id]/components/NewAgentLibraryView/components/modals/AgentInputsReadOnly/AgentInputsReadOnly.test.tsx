import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { AgentInputsReadOnly } from "./AgentInputsReadOnly";

function agent(inputProperties: Record<string, any>) {
  return {
    input_schema: { properties: inputProperties },
    credentials_input_schema: { properties: {} },
  } as unknown as LibraryAgent;
}

test("says there is no input when nothing the run stored is a graph input", () => {
  // Runs from before triggered presets nested their config stored the flat
  // trigger config as run inputs; no graph input schema describes those, and
  // the panel used to render an inputs card with nothing inside it.
  render(
    <AgentInputsReadOnly
      agent={agent({})}
      inputs={{ repo: "owner/repo", events: ["push"] }}
    />,
  );

  expect(screen.getByText("No input for this run.")).toBeDefined();
});

test("still renders the inputs the graph does declare", () => {
  render(
    <AgentInputsReadOnly
      agent={agent({ topic: { type: "string", title: "Topic" } })}
      inputs={{ topic: "weather", repo: "owner/repo" }}
    />,
  );

  expect(screen.queryByText("No input for this run.")).toBeNull();
});
