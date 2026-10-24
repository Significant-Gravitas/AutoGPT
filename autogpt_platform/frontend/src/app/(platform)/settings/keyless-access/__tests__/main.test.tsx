import { describe, expect, test } from "vitest";

import {
  getDeleteV1RevokeAWorkloadIdentityMockHandler200,
  getGetV1ListWorkloadIdentitiesMockHandler,
  getPatchV1EnableOrDisableAWorkloadIdentityMockHandler200,
} from "@/app/api/__generated__/endpoints/workload-identity/workload-identity.msw";
import { WorkloadIdentityStatus } from "@/app/api/__generated__/models/workloadIdentityStatus";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";

import SettingsKeylessAccessPage from "../page";
import { makeIdentity } from "./fixtures";

describe("SettingsKeylessAccessPage", () => {
  test("teaches the API key comparison even before anything is trusted", async () => {
    server.use(getGetV1ListWorkloadIdentitiesMockHandler([]));

    render(<SettingsKeylessAccessPage />);

    expect(await screen.findByText(/no trusted workloads yet/i)).toBeDefined();
    expect(screen.getByText(/how it works/i)).toBeDefined();
    expect(screen.getByText(/X-API-Key: \$AUTOGPT_API_KEY/)).toBeDefined();
    expect(
      screen.getByText(/Authorization: Bearer \$AUTOGPT_TOKEN/),
    ).toBeDefined();
  });

  test("lists trusted workloads with what they trust and when they were used", async () => {
    server.use(
      getGetV1ListWorkloadIdentitiesMockHandler([
        makeIdentity(),
        makeIdentity({
          id: "wi-revoked",
          name: "Old pipeline",
          status: WorkloadIdentityStatus.REVOKED,
        }),
      ]),
    );

    render(<SettingsKeylessAccessPage />);

    expect(await screen.findByText("GitHub Actions · acme/api")).toBeDefined();
    expect(
      screen.getByText("GitHub Actions · repository = acme/api"),
    ).toBeDefined();
    expect(screen.getByText("Never used")).toBeDefined();
    expect(screen.queryByText("Old pipeline")).toBeNull();
  });

  test("shows the curl walkthrough for a trusted workload", async () => {
    server.use(getGetV1ListWorkloadIdentitiesMockHandler([makeIdentity()]));

    render(<SettingsKeylessAccessPage />);
    fireEvent.click(await screen.findByRole("button", { name: /how to use/i }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/id-token: write/)).toBeDefined();
    expect(within(dialog).getByText(/WORKLOAD_ID="wi-123"/)).toBeDefined();
    expect(within(dialog).getByText(/\/api\/oauth\/token/)).toBeDefined();
    expect(
      within(dialog).getAllByText(/Authorization: Bearer \$AUTOGPT_TOKEN/)
        .length,
    ).toBeGreaterThan(0);
  });

  test("disables a workload", async () => {
    let sentStatus: unknown;
    server.use(
      getGetV1ListWorkloadIdentitiesMockHandler([makeIdentity()]),
      getPatchV1EnableOrDisableAWorkloadIdentityMockHandler200(async (info) => {
        sentStatus = await info.request.json();
        return makeIdentity({ status: WorkloadIdentityStatus.DISABLED });
      }),
    );

    render(<SettingsKeylessAccessPage />);
    fireEvent.click(
      await screen.findByRole("button", {
        name: /disable github actions · acme\/api/i,
      }),
    );

    await waitFor(() => {
      expect(sentStatus).toEqual({ status: "DISABLED" });
    });
  });

  test("revokes a workload after confirmation", async () => {
    let revokedPath = "";
    server.use(
      getGetV1ListWorkloadIdentitiesMockHandler([makeIdentity()]),
      getDeleteV1RevokeAWorkloadIdentityMockHandler200(async (info) => {
        revokedPath = new URL(info.request.url).pathname;
        return makeIdentity({ status: WorkloadIdentityStatus.REVOKED });
      }),
    );

    render(<SettingsKeylessAccessPage />);
    fireEvent.click(
      await screen.findByRole("button", {
        name: /revoke github actions · acme\/api/i,
      }),
    );
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: /^revoke$/i }));

    await waitFor(() => {
      expect(revokedPath).toMatch(/\/workload-identities\/wi-123$/);
    });
  });
});
