import { describe, it, expect, beforeEach } from "vitest";
import { useGraphStore } from "../stores/graphStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";

function createTestGraphMeta(
  overrides: Partial<GraphMeta> & { id: string; name: string },
): GraphMeta {
  return {
    version: 1,
    description: "",
    is_active: true,
    user_id: "test-user",
    created_at: new Date("2024-01-01T00:00:00Z"),
    ...overrides,
  };
}

function resetStore() {
  useGraphStore.setState({
    graphExecutionStatus: undefined,
    isGraphRunning: false,
    inputSchema: null,
    credentialsInputSchema: null,
    outputSchema: null,
    availableSubGraphs: [],
  });
}

beforeEach(() => {
  resetStore();
});

describe("graphStore", () => {
  describe("execution status transitions", () => {
    it("handles QUEUED -> RUNNING -> COMPLETED transition", () => {
      const { setGraphExecutionStatus } = useGraphStore.getState();

      setGraphExecutionStatus(AgentExecutionStatus.QUEUED);
      expect(useGraphStore.getState().graphExecutionStatus).toBe(
        AgentExecutionStatus.QUEUED,
      );
      expect(useGraphStore.getState().isGraphRunning).toBe(true);

      setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      expect(useGraphStore.getState().graphExecutionStatus).toBe(
        AgentExecutionStatus.RUNNING,
      );
      expect(useGraphStore.getState().isGraphRunning).toBe(true);

      setGraphExecutionStatus(AgentExecutionStatus.COMPLETED);
      expect(useGraphStore.getState().graphExecutionStatus).toBe(
        AgentExecutionStatus.COMPLETED,
      );
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });

    it("handles QUEUED -> RUNNING -> FAILED transition", () => {
      const { setGraphExecutionStatus } = useGraphStore.getState();

      setGraphExecutionStatus(AgentExecutionStatus.QUEUED);
      expect(useGraphStore.getState().isGraphRunning).toBe(true);

      setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      expect(useGraphStore.getState().isGraphRunning).toBe(true);

      setGraphExecutionStatus(AgentExecutionStatus.FAILED);
      expect(useGraphStore.getState().graphExecutionStatus).toBe(
        AgentExecutionStatus.FAILED,
      );
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });
  });

  describe("setGraphExecutionStatus auto-sets isGraphRunning", () => {
    it("sets isGraphRunning to true for RUNNING", () => {
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      expect(useGraphStore.getState().isGraphRunning).toBe(true);
    });

    it("sets isGraphRunning to true for QUEUED", () => {
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.QUEUED);
      expect(useGraphStore.getState().isGraphRunning).toBe(true);
    });

    it("sets isGraphRunning to false for COMPLETED", () => {
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      expect(useGraphStore.getState().isGraphRunning).toBe(true);

      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.COMPLETED);
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });

    it("sets isGraphRunning to false for FAILED", () => {
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.FAILED);
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });

    it("sets isGraphRunning to false for TERMINATED", () => {
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.TERMINATED);
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });

    it("sets isGraphRunning to false for INCOMPLETE", () => {
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.INCOMPLETE);
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });

    it("sets isGraphRunning to false for undefined", () => {
      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      expect(useGraphStore.getState().isGraphRunning).toBe(true);

      useGraphStore.getState().setGraphExecutionStatus(undefined);
      expect(useGraphStore.getState().graphExecutionStatus).toBeUndefined();
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });
  });

  describe("setIsGraphRunning", () => {
    it("sets isGraphRunning independently of status", () => {
      useGraphStore.getState().setIsGraphRunning(true);
      expect(useGraphStore.getState().isGraphRunning).toBe(true);

      useGraphStore.getState().setIsGraphRunning(false);
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    });
  });

  describe("schema management", () => {
    it("sets all three schemas via setGraphSchemas", () => {
      const input = { properties: { prompt: { type: "string" } } };
      const credentials = { properties: { apiKey: { type: "string" } } };
      const output = { properties: { result: { type: "string" } } };

      useGraphStore.getState().setGraphSchemas(input, credentials, output);

      const state = useGraphStore.getState();
      expect(state.inputSchema).toEqual(input);
      expect(state.credentialsInputSchema).toEqual(credentials);
      expect(state.outputSchema).toEqual(output);
    });

    it("sets schemas to null", () => {
      const input = { properties: { prompt: { type: "string" } } };
      useGraphStore.getState().setGraphSchemas(input, null, null);

      const state = useGraphStore.getState();
      expect(state.inputSchema).toEqual(input);
      expect(state.credentialsInputSchema).toBeNull();
      expect(state.outputSchema).toBeNull();
    });

    it("overwrites previous schemas", () => {
      const first = { properties: { a: { type: "string" } } };
      const second = { properties: { b: { type: "number" } } };

      useGraphStore.getState().setGraphSchemas(first, first, first);
      useGraphStore.getState().setGraphSchemas(second, null, second);

      const state = useGraphStore.getState();
      expect(state.inputSchema).toEqual(second);
      expect(state.credentialsInputSchema).toBeNull();
      expect(state.outputSchema).toEqual(second);
    });
  });

  describe("hasInputs", () => {
    it("returns false when inputSchema is null", () => {
      expect(useGraphStore.getState().hasInputs()).toBe(false);
    });

    it("returns false when inputSchema has no properties", () => {
      useGraphStore.getState().setGraphSchemas({}, null, null);
      expect(useGraphStore.getState().hasInputs()).toBe(false);
    });

    it("returns false when inputSchema has empty properties", () => {
      useGraphStore.getState().setGraphSchemas({ properties: {} }, null, null);
      expect(useGraphStore.getState().hasInputs()).toBe(false);
    });

    it("returns true when inputSchema has properties", () => {
      useGraphStore
        .getState()
        .setGraphSchemas(
          { properties: { prompt: { type: "string" } } },
          null,
          null,
        );
      expect(useGraphStore.getState().hasInputs()).toBe(true);
    });
  });

  describe("hasCredentials", () => {
    it("returns false when credentialsInputSchema is null", () => {
      expect(useGraphStore.getState().hasCredentials()).toBe(false);
    });

    it("returns false when credentialsInputSchema has empty properties", () => {
      useGraphStore.getState().setGraphSchemas(null, { properties: {} }, null);
      expect(useGraphStore.getState().hasCredentials()).toBe(false);
    });

    it("returns true when credentialsInputSchema has properties", () => {
      useGraphStore
        .getState()
        .setGraphSchemas(
          null,
          { properties: { apiKey: { type: "string" } } },
          null,
        );
      expect(useGraphStore.getState().hasCredentials()).toBe(true);
    });
  });

  describe("hasOutputs", () => {
    it("returns false when outputSchema is null", () => {
      expect(useGraphStore.getState().hasOutputs()).toBe(false);
    });

    it("returns false when outputSchema has empty properties", () => {
      useGraphStore.getState().setGraphSchemas(null, null, { properties: {} });
      expect(useGraphStore.getState().hasOutputs()).toBe(false);
    });

    it("returns true when outputSchema has properties", () => {
      useGraphStore.getState().setGraphSchemas(null, null, {
        properties: { result: { type: "string" } },
      });
      expect(useGraphStore.getState().hasOutputs()).toBe(true);
    });
  });

  describe("reset", () => {
    it("clears execution status and schemas but preserves outputSchema and availableSubGraphs", () => {
      const subGraphs: GraphMeta[] = [
        createTestGraphMeta({
          id: "sub-1",
          name: "Sub Graph",
          description: "A sub graph",
        }),
      ];

      useGraphStore
        .getState()
        .setGraphExecutionStatus(AgentExecutionStatus.RUNNING);
      useGraphStore
        .getState()
        .setGraphSchemas(
          { properties: { a: {} } },
          { properties: { b: {} } },
          { properties: { c: {} } },
        );
      useGraphStore.getState().setAvailableSubGraphs(subGraphs);

      useGraphStore.getState().reset();

      const state = useGraphStore.getState();
      expect(state.graphExecutionStatus).toBeUndefined();
      expect(state.isGraphRunning).toBe(false);
      expect(state.inputSchema).toBeNull();
      expect(state.credentialsInputSchema).toBeNull();
      // reset does not clear outputSchema or availableSubGraphs
      expect(state.outputSchema).toEqual({ properties: { c: {} } });
      expect(state.availableSubGraphs).toEqual(subGraphs);
    });

    it("is idempotent on fresh state", () => {
      useGraphStore.getState().reset();

      const state = useGraphStore.getState();
      expect(state.graphExecutionStatus).toBeUndefined();
      expect(state.isGraphRunning).toBe(false);
      expect(state.inputSchema).toBeNull();
      expect(state.credentialsInputSchema).toBeNull();
    });
  });

  describe("setAvailableSubGraphs", () => {
    it("sets sub-graphs list", () => {
      const graphs: GraphMeta[] = [
        createTestGraphMeta({
          id: "graph-1",
          name: "Graph One",
          description: "First graph",
        }),
        createTestGraphMeta({
          id: "graph-2",
          version: 2,
          name: "Graph Two",
          description: "Second graph",
        }),
      ];

      useGraphStore.getState().setAvailableSubGraphs(graphs);
      expect(useGraphStore.getState().availableSubGraphs).toEqual(graphs);
    });

    it("replaces previous sub-graphs", () => {
      const first: GraphMeta[] = [createTestGraphMeta({ id: "a", name: "A" })];
      const second: GraphMeta[] = [
        createTestGraphMeta({ id: "b", name: "B" }),
        createTestGraphMeta({ id: "c", name: "C" }),
      ];

      useGraphStore.getState().setAvailableSubGraphs(first);
      expect(useGraphStore.getState().availableSubGraphs).toHaveLength(1);

      useGraphStore.getState().setAvailableSubGraphs(second);
      expect(useGraphStore.getState().availableSubGraphs).toHaveLength(2);
      expect(useGraphStore.getState().availableSubGraphs).toEqual(second);
    });

    it("can set empty sub-graphs list", () => {
      useGraphStore
        .getState()
        .setAvailableSubGraphs([createTestGraphMeta({ id: "x", name: "X" })]);
      useGraphStore.getState().setAvailableSubGraphs([]);
      expect(useGraphStore.getState().availableSubGraphs).toEqual([]);
    });
  });
});
