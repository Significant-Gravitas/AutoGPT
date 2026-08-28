import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const push = vi.fn();
const toast = vi.fn();
const uploadFileDirect = vi.fn();
const fetchWorkflowFromUrl = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast }),
}));

vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: (file: File) => uploadFileDirect(file),
}));

vi.mock("../fetchWorkflowFromUrl", () => ({
  fetchWorkflowFromUrl: (url: string) => fetchWorkflowFromUrl(url),
}));

import { useExternalWorkflowTab } from "../useExternalWorkflowTab";

beforeEach(() => {
  push.mockReset();
  toast.mockReset();
  uploadFileDirect.mockReset();
  fetchWorkflowFromUrl.mockReset();
  sessionStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useExternalWorkflowTab", () => {
  it("imports on a LAN HTTP origin without crypto.randomUUID", async () => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal("crypto", {
      getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto),
    });
    fetchWorkflowFromUrl.mockResolvedValue({
      ok: true,
      json: JSON.stringify({ nodes: [] }),
    });
    uploadFileDirect.mockResolvedValue({
      file_id: "file-1",
      name: "workflow.json",
      mime_type: "application/json",
    });
    const { result } = renderHook(() => useExternalWorkflowTab());

    act(() => {
      result.current.setUrlValue("https://n8n.io/workflows/6270");
    });
    await act(async () => {
      await result.current.submitWithMode("url");
    });

    const file = uploadFileDirect.mock.calls[0][0] as File;
    expect(file.name).toMatch(
      /^workflow-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.json$/i,
    );
    expect(push).toHaveBeenCalledWith(
      "/copilot?organizationId=__personal__&teamId=__org_home__&source=import&autosubmit=true",
    );
    expect(toast).not.toHaveBeenCalled();
  });
});
