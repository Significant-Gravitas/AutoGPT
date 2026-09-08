import { getGetV1ListProvidersQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { getGetV1ListProvidersMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, useQueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../../../store";
import { ConnectionPicker } from "../ConnectionPicker/ConnectionPicker";
import {
  availableDeploymentOffer,
  mockMaxUpgrade,
  openPicker,
} from "./maxUpgradeFixtures";

const codex: ProviderMetadata = {
  name: "codex",
  supported_auth_types: ["oauth2"],
};
const connectName = "Connect a ChatGPT subscription";
const providersKey = getGetV1ListProvidersQueryKey();

beforeEach(() => {
  useCopilotUIStore.setState({
    copilotLlmAuth: null,
    copilotLlmModel: "standard",
  });
  mockMaxUpgrade([availableDeploymentOffer()]);
});

function renderPicker() {
  let client: QueryClient | undefined;
  function CaptureClient() {
    client = useQueryClient();
    return null;
  }
  render(
    <>
      <ConnectionPicker />
      <CaptureClient />
    </>,
  );
  return client!;
}

function expectNoChatGPTAction() {
  expect(screen.queryByRole("button", { name: connectName })).toBeNull();
  expect(screen.queryByText("Connect ChatGPT")).toBeNull();
  expect(screen.queryByRole("link", { name: "Upgrade to Max" })).toBeNull();
}

async function selectAdvanced() {
  await userEvent.click(
    screen.getByRole("radio", { name: "Advanced · opus-server" }),
  );
  expect(useCopilotUIStore.getState().copilotLlmModel).toBe("advanced");
}

describe("ChatGPT connection eligibility", () => {
  it("keeps model tiers usable without a connection or upsell when the provider is unavailable", async () => {
    server.use(getGetV1ListProvidersMockHandler200([]));
    const client = renderPicker();
    await openPicker();
    expectNoChatGPTAction();
    await waitFor(() =>
      expect(client.getQueryState(providersKey)?.status).toBe("success"),
    );
    expectNoChatGPTAction();
    await selectAdvanced();
  });

  it("shows the normal Connect action when the server offers the codex provider", async () => {
    server.use(getGetV1ListProvidersMockHandler200([codex]));
    const client = renderPicker();
    await openPicker();
    await waitFor(() =>
      expect(client.getQueryState(providersKey)?.status).toBe("success"),
    );
    expect(
      (await screen.findByRole("button", { name: connectName })).isConnected,
    ).toBe(true);
    expect(screen.queryByText("Connect ChatGPT")).toBeNull();
    await selectAdvanced();
  });

  it("waits for a pending eligibility response before showing Connect", async () => {
    let resolveProviders!: (providers: ProviderMetadata[]) => void;
    const providers = new Promise<ProviderMetadata[]>((resolve) => {
      resolveProviders = resolve;
    });
    server.use(getGetV1ListProvidersMockHandler200(() => providers));
    const client = renderPicker();
    try {
      await openPicker();
      expectNoChatGPTAction();
      await waitFor(() =>
        expect(client.getQueryState(providersKey)?.fetchStatus).toBe(
          "fetching",
        ),
      );
      expectNoChatGPTAction();
      resolveProviders([codex]);
      expect(
        (await screen.findByRole("button", { name: connectName })).isConnected,
      ).toBe(true);
    } finally {
      resolveProviders([]);
    }
  });

  it("keeps models usable without inventing an upsell when eligibility fails", async () => {
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 500 }),
      ),
    );
    const client = renderPicker();
    await openPicker();
    expectNoChatGPTAction();
    await waitFor(() =>
      expect(client.getQueryState(providersKey)?.status).toBe("error"),
    );
    expectNoChatGPTAction();
    await selectAdvanced();
  });

  it("removes Connect when a previously successful eligibility check fails", async () => {
    server.use(getGetV1ListProvidersMockHandler200([codex]));
    const client = renderPicker();
    await openPicker();
    await screen.findByRole("button", { name: connectName });
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 500 }),
      ),
    );
    await client.refetchQueries({ queryKey: providersKey });
    await waitFor(() =>
      expect(client.getQueryState(providersKey)?.status).toBe("error"),
    );
    await waitFor(expectNoChatGPTAction);
    await selectAdvanced();
  });
});
