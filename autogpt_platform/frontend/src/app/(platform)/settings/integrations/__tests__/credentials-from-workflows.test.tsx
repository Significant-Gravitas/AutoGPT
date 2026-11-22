import { QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { useState } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import type {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { server } from "@/mocks/mock-server";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";

import SettingsIntegrationsPage from "../page";

// A key added or deleted in a workflow step's credentials field, the builder's
// or the run dialog's, is the same key Settings → Integrations lists. The app
// keeps that list fresh for a minute, and navigating back to the page reads
// it from the cache, so the field has to tell the list it changed.

// CredentialsProvider only fetches for a signed-in user; the shared setup
// signs everyone out.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false, user: null }),
}));

const PROVIDER = "unreal_speech";

const SPEECH_FIELD = {
  type: "object",
  title: "Credentials",
  properties: {},
  required: ["id", "provider", "type"],
  credentials_provider: [PROVIDER],
  credentials_types: ["api_key"],
} as unknown as BlockIOCredentialsSubSchema;

const EXISTING_KEY: CredentialsMetaResponse = {
  id: "cred-existing",
  provider: PROVIDER,
  type: "api_key",
  title: "Studio key",
  scopes: null,
  username: null,
  host: null,
  is_managed: false,
};

let stored: CredentialsMetaResponse[] = [];

function serveCredentials() {
  server.use(
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([
        {
          name: PROVIDER,
          description: "Text to speech",
          supported_auth_types: ["api_key"],
        },
      ]),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json([]),
    ),
    http.get("*/api/integrations/credentials", () => HttpResponse.json(stored)),
    http.get(`*/api/integrations/${PROVIDER}/credentials`, () =>
      HttpResponse.json(stored),
    ),
    http.post(
      `*/api/integrations/${PROVIDER}/credentials`,
      async ({ request }) => {
        const body = (await request.json()) as { title: string };
        const created: CredentialsMetaResponse = {
          ...EXISTING_KEY,
          id: "cred-added",
          title: body.title,
        };
        stored = [...stored, created];
        return HttpResponse.json(created, { status: 201 });
      },
    ),
    http.delete(
      `*/api/integrations/${PROVIDER}/credentials/:id`,
      ({ params }) => {
        stored = stored.filter((credential) => credential.id !== params.id);
        return HttpResponse.json({ deleted: true, revoked: null });
      },
    ),
  );
}

// The app's own query defaults: a list read in the last minute is fresh.
function appQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { staleTime: 60 * 1000, retry: false } },
  });
}

function WorkflowStepCredentials() {
  const [selected, setSelected] = useState<CredentialsMetaInput>();
  return (
    <CredentialsInput
      schema={SPEECH_FIELD}
      selectedCredentials={selected}
      onSelectCredentials={setSelected}
      variant="node"
    />
  );
}

function App({ page }: { page: "integrations" | "workflow" }) {
  return (
    <CredentialsProvider>
      {page === "integrations" ? (
        <SettingsIntegrationsPage />
      ) : (
        <WorkflowStepCredentials />
      )}
    </CredentialsProvider>
  );
}

function findListedKey(title: string) {
  return screen.findByRole("checkbox", { name: `Select ${title}` });
}

// The field's row for a saved key carries one unlabelled button: its menu.
function rowMenuButton(keyName: HTMLElement) {
  let row: HTMLElement | null = keyName;
  while (row && !row.querySelector("button")) row = row.parentElement;
  return within(row!).getByRole("button");
}

beforeEach(() => {
  stored = [EXISTING_KEY];
  serveCredentials();
});

describe("Settings → Integrations after a workflow step's credentials change", () => {
  test("lists a key added from a workflow step's credentials field", async () => {
    const queryClient = appQueryClient();
    const view = render(<App page="integrations" />, { queryClient });
    await findListedKey("Studio key");

    view.rerender(<App page="workflow" />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Use a new API key" }),
    );
    fireEvent.change(
      await screen.findByPlaceholderText("Enter a name for this API Key..."),
      { target: { value: "Narration key" } },
    );
    fireEvent.change(screen.getByPlaceholderText("Enter API Key..."), {
      target: { value: "not-a-real-key" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add API Key" }));
    await waitFor(() =>
      expect(stored.map((credential) => credential.title)).toContain(
        "Narration key",
      ),
    );
    await waitFor(() =>
      expect(screen.queryByPlaceholderText("Enter API Key...")).toBeNull(),
    );

    view.rerender(<App page="integrations" />);
    await findListedKey("Narration key");
    expect(
      screen.getByRole("checkbox", { name: "Select Studio key" }),
    ).toBeTruthy();
  });

  test("drops a key deleted from a workflow step's credentials field", async () => {
    const queryClient = appQueryClient();
    const view = render(<App page="integrations" />, { queryClient });
    await findListedKey("Studio key");

    view.rerender(<App page="workflow" />);
    fireEvent.pointerDown(
      rowMenuButton(await screen.findByText("Studio key")),
      {
        button: 0,
        ctrlKey: false,
      },
    );
    fireEvent.click(await screen.findByRole("menuitem", { name: "Delete" }));
    const confirm = await screen.findByRole("dialog");
    fireEvent.click(within(confirm).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(stored).toEqual([]));

    view.rerender(<App page="integrations" />);
    expect(await screen.findByText("No integration connected")).toBeTruthy();
    expect(
      screen.queryByRole("checkbox", { name: "Select Studio key" }),
    ).toBeNull();
  });
});
