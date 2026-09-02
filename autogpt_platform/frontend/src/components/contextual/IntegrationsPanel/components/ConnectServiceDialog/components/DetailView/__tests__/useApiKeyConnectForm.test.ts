import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useApiKeyConnectForm } from "../useApiKeyConnectForm";

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

const postCredentials = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  postV1CreateCredentials: (...args: unknown[]) => postCredentials(...args),
  getGetV1ListCredentialsQueryKey: vi.fn(() => ["credentials"]),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
}));

const VALUES = { title: "My token", apiKey: "sk-123", expiresAt: "" };

afterEach(() => {
  vi.clearAllMocks();
});

describe("useApiKeyConnectForm", () => {
  it("forwards credential metadata so an MCP key is tagged with its server", async () => {
    // Without `mcp_server_url` the credential comes back with `host: null`,
    // no picker can match it, and the block 401s with nothing explaining why.
    postCredentials.mockResolvedValue({ data: { id: "cred-1" } });
    const onSuccess = vi.fn();

    const { result } = renderHook(() =>
      useApiKeyConnectForm({
        provider: "mcp",
        onSuccess,
        metadata: { mcp_server_url: "https://mcp.datafa.st/mcp" },
      }),
    );

    await result.current.handleSubmit(VALUES);

    expect(postCredentials).toHaveBeenCalledWith(
      "mcp",
      expect.objectContaining({
        provider: "mcp",
        type: "api_key",
        api_key: "sk-123",
        metadata: { mcp_server_url: "https://mcp.datafa.st/mcp" },
      }),
    );
    await waitFor(() => expect(onSuccess).toHaveBeenCalledOnce());
  });

  it("omits the metadata key entirely when there is none", async () => {
    postCredentials.mockResolvedValue({ data: { id: "cred-1" } });

    const { result } = renderHook(() =>
      useApiKeyConnectForm({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.handleSubmit(VALUES);

    expect(postCredentials.mock.calls[0][1]).not.toHaveProperty("metadata");
  });

  it("converts an expiry date to unix seconds", async () => {
    postCredentials.mockResolvedValue({ data: { id: "cred-1" } });

    const { result } = renderHook(() =>
      useApiKeyConnectForm({ provider: "github", onSuccess: vi.fn() }),
    );

    await result.current.handleSubmit({
      ...VALUES,
      expiresAt: "2030-01-01T00:00:00.000Z",
    });

    expect(postCredentials.mock.calls[0][1].expires_at).toBe(
      Math.floor(Date.parse("2030-01-01T00:00:00.000Z") / 1000),
    );
  });

  it("does not report success when the request fails", async () => {
    postCredentials.mockRejectedValue(new Error("nope"));
    const onSuccess = vi.fn();

    const { result } = renderHook(() =>
      useApiKeyConnectForm({ provider: "github", onSuccess }),
    );

    await result.current.handleSubmit(VALUES);

    expect(onSuccess).not.toHaveBeenCalled();
  });
});
