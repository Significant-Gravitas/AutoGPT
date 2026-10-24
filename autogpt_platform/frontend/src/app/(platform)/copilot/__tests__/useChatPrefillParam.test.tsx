import { renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../store";
import { useChatPrefillParam } from "../useChatPrefillParam";

describe("useChatPrefillParam", () => {
  afterEach(() => useCopilotUIStore.getState().setInitialPrompt(null));

  it("drafts ?prefill= into the composer once", async () => {
    renderHook(() => useChatPrefillParam(), {
      wrapper: ({ children }) => (
        <NuqsTestingAdapter searchParams="?sessionId=sub-1&prefill=Q4%20release">
          {children}
        </NuqsTestingAdapter>
      ),
    });
    await waitFor(() =>
      expect(useCopilotUIStore.getState().initialPrompt).toBe("Q4 release"),
    );
  });
});
