import { describe, expect, test } from "vitest";

import {
  getGetV1ListWorkloadIdentitiesMockHandler,
  getPostV1TrustAWorkloadIdentityMockHandler200,
} from "@/app/api/__generated__/endpoints/workload-identity/workload-identity.msw";
import type { WorkloadIdentitySpec } from "@/app/api/__generated__/models/workloadIdentitySpec";
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

async function openCreateDialog() {
  render(<SettingsKeylessAccessPage />);
  await screen.findByText(/no trusted workloads yet/i);
  fireEvent.click(
    screen.getAllByRole("button", { name: /^add workload$/i })[0],
  );
  return screen.findByRole("dialog");
}

function captureCreate(response = makeIdentity({ id: "wi-new" })) {
  const sent: { body?: WorkloadIdentitySpec } = {};
  server.use(
    getGetV1ListWorkloadIdentitiesMockHandler([]),
    getPostV1TrustAWorkloadIdentityMockHandler200(async (info) => {
      sent.body = (await info.request.json()) as WorkloadIdentitySpec;
      return response;
    }),
  );
  return sent;
}

async function submit(dialog: HTMLElement) {
  const button = within(dialog).getByRole("button", {
    name: /trust this workload/i,
  }) as HTMLButtonElement;
  await waitFor(() => {
    expect(button.disabled).toBe(false);
  });
  fireEvent.click(button);
}

describe("SettingsKeylessAccessPage - provisioning", () => {
  test("trusts a GitHub repository and shows how to call the API from it", async () => {
    const sent = captureCreate();
    const dialog = await openCreateDialog();

    fireEvent.click(
      within(dialog).getByRole("button", { name: /github actions/i }),
    );
    fireEvent.change(within(dialog).getByLabelText(/^repository$/i), {
      target: { value: "acme/api" },
    });
    fireEvent.change(within(dialog).getByLabelText(/branch/i), {
      target: { value: "main" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /identity/i }),
    );
    await submit(dialog);

    expect(await screen.findByText(/workload trusted/i)).toBeDefined();
    expect(sent.body?.issuer).toBe(
      "https://token.actions.githubusercontent.com",
    );
    expect(sent.body?.claim_conditions).toEqual([
      { claim: "repository", pattern: "acme/api" },
      { claim: "ref", pattern: "refs/heads/main" },
    ]);
    expect(sent.body?.permissions).toEqual(["IDENTITY"]);
    expect(screen.getByText(/WORKLOAD_ID="wi-new"/)).toBeDefined();
    expect(screen.getAllByText(/\/api\/oauth\/token/).length).toBeGreaterThan(
      0,
    );
  });

  test("keeps the submit button disabled until the repository is valid", async () => {
    captureCreate();
    const dialog = await openCreateDialog();

    fireEvent.click(
      within(dialog).getByRole("button", { name: /github actions/i }),
    );
    fireEvent.change(within(dialog).getByLabelText(/^repository$/i), {
      target: { value: "just-a-name" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /identity/i }),
    );

    const button = within(dialog).getByRole("button", {
      name: /trust this workload/i,
    }) as HTMLButtonElement;
    expect(
      await within(dialog).findByText(/use owner\/repository/i),
    ).toBeDefined();
    expect(button.disabled).toBe(true);
  });

  test("makes your own site the issuer and walks through publishing its key", async () => {
    const sent = captureCreate(
      makeIdentity({
        id: "wi-site",
        issuer: "https://acme.com",
        claim_conditions: [{ claim: "sub", pattern: "backend" }],
      }),
    );
    const dialog = await openCreateDialog();

    fireEvent.click(
      within(dialog).getByRole("button", { name: /your own website/i }),
    );
    fireEvent.change(within(dialog).getByLabelText(/your site/i), {
      target: { value: "https://acme.com/" },
    });
    fireEvent.change(within(dialog).getByLabelText(/server name/i), {
      target: { value: "backend" },
    });
    fireEvent.click(
      within(dialog).getByRole("checkbox", { name: /identity/i }),
    );
    await submit(dialog);

    expect(await screen.findByText(/workload trusted/i)).toBeDefined();
    expect(sent.body?.issuer).toBe("https://acme.com");
    expect(sent.body?.claim_conditions).toEqual([
      { claim: "sub", pattern: "backend" },
    ]);
    expect(
      screen.getByText(
        /https:\/\/acme\.com\/\.well-known\/openid-configuration/,
      ),
    ).toBeDefined();
    expect(screen.getByText(/\.setSubject\("backend"\)/)).toBeDefined();
  });
});
