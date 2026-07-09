import { server } from "@/mocks/mock-server";
import { useGetFlag } from "@/services/feature-flags/use-get-flag";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHANGELOG_INDEX_MD_URL } from "../changelog-constants";
import { ChangelogPopup } from "../ChangelogPopup";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: vi.fn() };
});

let indexRequests = 0;

beforeEach(() => {
  indexRequests = 0;
  // The toast fetches the changelog index on mount; count hits so we can prove
  // whether the toast actually mounted under each layout.
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => {
      indexRequests += 1;
      return HttpResponse.text("");
    }),
  );
});

afterEach(() => {
  server.resetHandlers();
  vi.mocked(useGetFlag).mockReset();
});

describe("ChangelogPopup", () => {
  it("renders nothing and skips the changelog fetch on the new sidebar layout", async () => {
    vi.mocked(useGetFlag).mockReturnValue(true);
    render(<ChangelogPopup />);

    // Give any effects a chance to run — none should, since the gate short
    // -circuits before the toast (and its fetch/auto-dismiss) ever mounts.
    await new Promise((resolve) => setTimeout(resolve, 25));
    expect(indexRequests).toBe(0);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("mounts the floating toast on the classic layout", async () => {
    vi.mocked(useGetFlag).mockReturnValue(false);
    render(<ChangelogPopup />);

    await waitFor(() => expect(indexRequests).toBeGreaterThan(0));
  });
});
