import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useExternalWorkflowTab } from "../useExternalWorkflowTab";

const { uploadFileDirectMock, pushMock } = vi.hoisted(() => ({
  uploadFileDirectMock: vi.fn(),
  pushMock: vi.fn(),
}));

vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: uploadFileDirectMock,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
  sessionStorage.clear();
});

// Chrome before 92 and plain-HTTP origins have no crypto.randomUUID.
describe("useExternalWorkflowTab where crypto.randomUUID is missing", () => {
  it("uploads a workflow file and opens Otto with it", async () => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal("crypto", {
      getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto),
    });
    uploadFileDirectMock.mockResolvedValue({
      file_id: "file-1",
      name: "workflow.json",
      mime_type: "application/json",
    });
    const { result } = renderHook(() => useExternalWorkflowTab());

    act(() => {
      result.current.setFileValue(
        `data:application/json;base64,${btoa('{"nodes":[]}')}`,
      );
    });
    await act(async () => {
      await result.current.submitWithMode("file");
    });

    const file: File = uploadFileDirectMock.mock.calls[0][0];
    expect(file.name).toMatch(/^workflow-[0-9a-f-]{36}\.json$/);
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?source=import&autosubmit=true",
    );
  });
});
