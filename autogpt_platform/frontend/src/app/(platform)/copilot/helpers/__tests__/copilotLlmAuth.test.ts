import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { describe, expect, it } from "vitest";
import { resolveCopilotLLMAuthSelection } from "../copilotLlmAuth";

function platform(isDefault: boolean): ChatTransportResponse {
  return {
    auth_provider: "platform",
    credential_id: null,
    label: "AutoGPT Platform",
    available: true,
    default: isDefault,
  };
}

function chatgpt(
  credentialId: string,
  isDefault: boolean,
): ChatTransportResponse {
  return {
    auth_provider: "codex",
    credential_id: credentialId,
    label: "ChatGPT",
    available: true,
    default: isDefault,
  };
}

describe("resolveCopilotLLMAuthSelection", () => {
  it("adopts the saved default when nothing has been chosen yet", () => {
    const resolved = resolveCopilotLLMAuthSelection(
      [platform(false), chatgpt("cred-1", true)],
      null,
    );

    expect(resolved).toEqual({
      authProvider: "codex",
      credentialId: "cred-1",
    });
  });

  it("leaves a usable choice alone, so a chat in progress keeps its connection", () => {
    const resolved = resolveCopilotLLMAuthSelection(
      [platform(false), chatgpt("cred-1", true)],
      { authProvider: "platform", credentialId: null },
    );

    expect(resolved).toEqual({
      authProvider: "platform",
      credentialId: null,
    });
  });

  it("keeps the saved default pointed at one account", () => {
    const resolved = resolveCopilotLLMAuthSelection(
      [chatgpt("cred-1", false), chatgpt("cred-2", true)],
      null,
    );

    expect(resolved).toEqual({
      authProvider: "codex",
      credentialId: "cred-2",
    });
  });

  it("falls back to the only connection there is", () => {
    const resolved = resolveCopilotLLMAuthSelection(
      [chatgpt("cred-1", false)],
      null,
    );

    expect(resolved).toEqual({
      authProvider: "codex",
      credentialId: "cred-1",
    });
  });

  it("asks rather than guesses when several connections and no default", () => {
    const resolved = resolveCopilotLLMAuthSelection(
      [chatgpt("cred-1", false), chatgpt("cred-2", false)],
      null,
    );

    expect(resolved).toBeNull();
  });

  it("waits for the transport list before choosing anything", () => {
    expect(resolveCopilotLLMAuthSelection(undefined, null)).toBeNull();
  });
});
