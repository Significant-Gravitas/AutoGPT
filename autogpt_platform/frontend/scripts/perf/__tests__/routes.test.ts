import { describe, expect, it } from "vitest";
import { matchRoute, type RouteHandlers } from "../lib/routes";

const handler = async () => undefined;
const all: RouteHandlers = {
  featureFlags: handler,
  onboardingCompleted: handler,
  chatStream: handler,
  chatSessionList: handler,
  chatSession: handler,
  chatSessionChild: handler,
  graphExecution: handler,
  graph: handler,
  libraryAgents: handler,
  executions: handler,
  libraryAgent: handler,
  graphExecutions: handler,
  graphSchedules: handler,
  libraryPresets: handler,
};

function nameOf(url: string, handlers: RouteHandlers = all) {
  return matchRoute(url, handlers)?.name ?? null;
}

describe("matchRoute", () => {
  it("routes the copilot stream on the backend host and through the proxy", () => {
    expect(nameOf("http://localhost:8006/api/chat/sessions/s1/stream")).toBe(
      "chatStream",
    );
    expect(
      nameOf("http://localhost:3000/api/proxy/api/chat/sessions/s1/stream"),
    ).toBe("chatStream");
  });

  it("tells the session list, a session and its sub-resources apart", () => {
    const base = "http://localhost:3000/api/proxy/api/chat/sessions";
    expect(nameOf(`${base}?limit=50&offset=0`)).toBe("chatSessionList");
    expect(nameOf(`${base}/s1`)).toBe("chatSession");
    expect(nameOf(`${base}/s1?limit=50&before_sequence=9`)).toBe("chatSession");
    const child = matchRoute(`${base}/s1/messages/pending`, all);
    expect(child?.name).toBe("chatSessionChild");
    expect(child?.match[2]).toBe("messages/pending");
  });

  it("matches an execution before its graph", () => {
    const base = "http://localhost:3000/api/proxy/api/graphs/g1";
    expect(nameOf(`${base}/executions/e1`)).toBe("graphExecution");
    expect(nameOf(`${base}?version=1`)).toBe("graph");
  });

  it("tells a workflow's runs, schedules and presets apart from one run", () => {
    const base = "http://localhost:3000/api/proxy/api";
    expect(nameOf(`${base}/graphs/g1/executions?page=1&page_size=20`)).toBe(
      "graphExecutions",
    );
    expect(nameOf(`${base}/graphs/g1/executions/e1`)).toBe("graphExecution");
    expect(nameOf(`${base}/graphs/g1/schedules`)).toBe("graphSchedules");
    expect(nameOf(`${base}/library/presets?graph_id=g1`)).toBe(
      "libraryPresets",
    );
    expect(nameOf(`${base}/library/agents/a1`)).toBe("libraryAgent");
    expect(nameOf(`${base}/library/agents?page_size=20`)).toBe("libraryAgents");
  });

  it("leaves pages, other endpoints and unhandled routes alone", () => {
    expect(nameOf("http://localhost:3000/home?sessionId=s1")).toBeNull();
    expect(
      nameOf("http://localhost:3000/api/proxy/api/executions/cost-summary"),
    ).toBeNull();
    expect(
      nameOf("http://localhost:3000/api/proxy/api/executions", {
        graph: handler,
      }),
    ).toBeNull();
  });
});
