import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { describe, expect, it } from "vitest";

import {
  getDeletedKeys,
  getRefusedKeys,
  getRerunForm,
  getRerunFormNotice,
} from "../helpers";

type RunCredentials = GraphExecution["credential_inputs"];

function key(id: string, provider: string, title?: string) {
  return { id, provider, type: "api_key" as const, title };
}

// The user's saved keys, as the credentials provider lists them
function listing(saved: Record<string, string[]>) {
  return Object.fromEntries(
    Object.entries(saved).map(([provider, ids]) => [
      provider,
      { savedCredentials: ids.map((id) => key(id, provider)) },
    ]),
  ) as unknown as CredentialsProvidersContextType;
}

describe("getDeletedKeys", () => {
  it("names a deleted key by its title, else by its provider", () => {
    const credentials: RunCredentials = {
      firecrawl: key("gone-1", "firecrawl", "My Firecrawl key"),
      exa: key("gone-2", "exa"),
      jina: key("kept", "jina", "My Jina key"),
    };

    expect(
      getDeletedKeys(
        credentials,
        listing({ firecrawl: [], exa: ["other"], jina: ["kept"] }),
      ),
    ).toEqual([
      { field: "firecrawl", name: "My Firecrawl key" },
      { field: "exa", name: "Your Exa account" },
    ]);
  });

  it("finds none while the saved keys are loading", () => {
    expect(
      getDeletedKeys({ firecrawl: key("gone", "firecrawl") }, null),
    ).toEqual([]);
  });

  it("leaves the platform's keys and unlisted providers to the server", () => {
    const credentials: RunCredentials = {
      openai: key("53c25cb8", "openai", "Use Credits for OpenAI"),
      firecrawl: key("firecrawl-default", "firecrawl", "Firecrawl API Key"),
      retired: key("gone", "retired_provider", "Old key"),
      inherited: key("gone", "constructor", "Odd key"),
    };

    expect(
      getDeletedKeys(credentials, listing({ openai: [], firecrawl: [] })),
    ).toEqual([]);
  });
});

describe("getRefusedKeys", () => {
  function refusal(nodeErrors: Record<string, Record<string, string>>) {
    return new ApiError("Workflow validation failed", 400, {
      detail: {
        type: "validation_error",
        message: "Workflow validation failed",
        node_errors: nodeErrors,
      },
    });
  }

  it("finds the run's own keys the server doesn't know", () => {
    const credentials: RunCredentials = {
      firecrawl: key("gone", "firecrawl", "My Firecrawl key"),
      openai: key("53c25cb8", "openai", "Use Credits for OpenAI"),
    };
    const error = refusal({
      "step-1": { credentials: "Unknown credentials #gone" },
      "step-2": { credentials: "Unknown credentials #53c25cb8" },
    });

    expect(getRefusedKeys(error, credentials)).toEqual([
      { field: "firecrawl", name: "My Firecrawl key" },
    ]);
  });

  it("finds none for other refusals", () => {
    const credentials: RunCredentials = {
      firecrawl: key("expired", "firecrawl", "My Firecrawl key"),
    };

    expect(
      getRefusedKeys(
        refusal({ "step-1": { credentials: "Credentials not available" } }),
        credentials,
      ),
    ).toEqual([]);
    expect(getRefusedKeys(new Error("offline"), credentials)).toEqual([]);
  });
});

describe("getRerunForm", () => {
  it("keeps the run's inputs and its keys that still exist", () => {
    const run = {
      inputs: { topic: "news" },
      credential_inputs: {
        firecrawl: key("gone", "firecrawl", "My Firecrawl key"),
        jina: key("kept", "jina", "My Jina key"),
      },
    };

    expect(
      getRerunForm(run, [{ field: "firecrawl", name: "My Firecrawl key" }]),
    ).toEqual({
      deletedKeys: [{ field: "firecrawl", name: "My Firecrawl key" }],
      inputs: { topic: "news" },
      credentials: { jina: key("kept", "jina", "My Jina key") },
      fieldsToPick: ["firecrawl"],
    });
  });
});

describe("getRerunFormNotice", () => {
  it("names each deleted key once", () => {
    expect(
      getRerunFormNotice([
        { field: "firecrawl", name: "My Firecrawl key" },
        { field: "exa", name: "My Exa key" },
        { field: "exa_search", name: "My Exa key" },
      ]),
    ).toBe(
      "Keys this task used were deleted: My Firecrawl key and My Exa key. Choose other keys below to run it again.",
    );
  });
});
