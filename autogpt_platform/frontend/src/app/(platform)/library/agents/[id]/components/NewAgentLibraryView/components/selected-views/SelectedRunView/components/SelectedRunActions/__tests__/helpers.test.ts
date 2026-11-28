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

// A saved key, the platform's when `is_system` says so
type Saved = string | { id: string; is_system: boolean };

// The user's saved keys, as the credentials provider lists them
function listing(saved: Record<string, Saved[]>) {
  return Object.fromEntries(
    Object.entries(saved).map(([provider, keys]) => [
      provider,
      {
        savedCredentials: keys.map((entry) => {
          const { id, is_system } =
            typeof entry === "string" ? { id: entry, is_system: false } : entry;
          return { ...key(id, provider), is_system };
        }),
      },
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

  it("calls a deleted key of the user's deleted, whatever its title says", () => {
    const credentials: RunCredentials = {
      firecrawl: key("gone-1", "firecrawl", "Production system key"),
      exa: key("gone-2", "exa", "Use Credits for the team"),
    };

    expect(
      getDeletedKeys(credentials, listing({ firecrawl: ["other"], exa: [] })),
    ).toEqual([
      { field: "firecrawl", name: "Production system key" },
      { field: "exa", name: "Use Credits for the team" },
    ]);
  });

  it("leaves a provider's own SDK key and unlisted providers to the server", () => {
    // "<provider>-default" names the platform's SDK key, an id the server
    // never gives a user's key
    const credentials: RunCredentials = {
      firecrawl: key("firecrawl-default", "firecrawl", "Firecrawl API Key"),
      retired: key("gone", "retired_provider", "Old key"),
      inherited: key("gone", "constructor", "Odd key"),
    };

    expect(getDeletedKeys(credentials, listing({ firecrawl: [] }))).toEqual([]);
  });

  it("takes a built-in platform key the page no longer lists for a deleted one: only the saved keys mark its id as the platform's", () => {
    // The platform stopped offering it, so the server would refuse it too
    const credentials: RunCredentials = {
      openai: key("53c25cb8", "openai", "Use Credits for OpenAI"),
    };

    expect(getDeletedKeys(credentials, listing({ openai: [] }))).toEqual([
      { field: "openai", name: "Use Credits for OpenAI" },
    ]);
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

  it("finds the run's own keys the server doesn't know, not the platform's", () => {
    const credentials: RunCredentials = {
      firecrawl: key("gone", "firecrawl", "My Firecrawl key"),
      openai: key("53c25cb8", "openai", "OpenAI"),
      exa: key("exa-default", "exa", "Exa API Key"),
    };
    const error = refusal({
      "step-1": { credentials: "Unknown credentials #gone" },
      "step-2": { credentials: "Unknown credentials #53c25cb8" },
      "step-3": { credentials: "Unknown credentials #exa-default" },
    });

    expect(
      getRefusedKeys(
        error,
        credentials,
        listing({
          firecrawl: ["gone"],
          openai: [{ id: "53c25cb8", is_system: true }],
        }),
      ),
    ).toEqual([{ field: "firecrawl", name: "My Firecrawl key" }]);
  });

  it("calls a refused key of the user's deleted, whatever its title says", () => {
    const credentials: RunCredentials = {
      firecrawl: key("gone", "firecrawl", "Production system key"),
    };

    expect(
      getRefusedKeys(
        refusal({ "step-1": { credentials: "Unknown credentials #gone" } }),
        credentials,
        listing({ firecrawl: ["gone"] }),
      ),
    ).toEqual([{ field: "firecrawl", name: "Production system key" }]);
  });

  it("finds none for other refusals", () => {
    const credentials: RunCredentials = {
      firecrawl: key("expired", "firecrawl", "My Firecrawl key"),
    };
    const saved = listing({ firecrawl: ["expired"] });

    expect(
      getRefusedKeys(
        refusal({ "step-1": { credentials: "Credentials not available" } }),
        credentials,
        saved,
      ),
    ).toEqual([]);
    expect(getRefusedKeys(new Error("offline"), credentials, saved)).toEqual(
      [],
    );
  });

  it("finds none while the saved keys are loading, as only they mark the platform's ids", () => {
    const credentials: RunCredentials = {
      openai: key("53c25cb8", "openai", "OpenAI"),
    };

    expect(
      getRefusedKeys(
        refusal({ "step-1": { credentials: "Unknown credentials #53c25cb8" } }),
        credentials,
        null,
      ),
    ).toEqual([]);
  });
});

describe("getRerunForm", () => {
  it("keeps the run's inputs and its keys that still exist, and leaves the deleted ones' fields to pick from their providers", () => {
    const run = {
      id: "past-run",
      inputs: { topic: "news" },
      credential_inputs: {
        llm: key("gone", "openai", "My OpenAI key"),
        jina: key("kept", "jina", "My Jina key"),
      },
    };

    expect(
      getRerunForm(run, [{ field: "llm", name: "My OpenAI key" }]),
    ).toEqual({
      runId: "past-run",
      deletedKeys: [{ field: "llm", name: "My OpenAI key" }],
      inputs: { topic: "news" },
      credentials: { jina: key("kept", "jina", "My Jina key") },
      fieldsToPick: { llm: "openai" },
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
