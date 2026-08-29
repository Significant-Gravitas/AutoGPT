import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  getDeleteV1RevokeApiKeyMockHandler,
  getGetV1ListUserApiKeysMockHandler,
  getPostV1CreateNewApiKeyMockHandler,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";
import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import { APIKeyStatus } from "@/app/api/__generated__/models/aPIKeyStatus";
import { server } from "@/mocks/mock-server";
import ApiKeysPage from "../page";
import { beforeEach, describe, expect, test } from "vitest";

type ApiKeyRecord = {
  id: string;
  name: string;
  head: string;
  tail: string;
  status: APIKeyStatus;
};

function toApiKeyResponse(key: ApiKeyRecord) {
  return {
    id: key.id,
    user_id: "user-1",
    scopes: [APIKeyPermission.EXECUTE_GRAPH],
    type: "api_key" as const,
    created_at: new Date("2026-01-01T00:00:00.000Z"),
    expires_at: null,
    last_used_at: null,
    revoked_at: null,
    name: key.name,
    head: key.head,
    tail: key.tail,
    status: key.status,
    description: null,
  };
}

describe("ApiKeysPage", () => {
  let apiKeys: ApiKeyRecord[];
  let revokedKeyId: string;

  beforeEach(() => {
    apiKeys = [];
    revokedKeyId = "";

    server.use(
      getGetV1ListUserApiKeysMockHandler(() =>
        apiKeys.map((key) => toApiKeyResponse(key)),
      ),
      getPostV1CreateNewApiKeyMockHandler(async ({ request }) => {
        const body = (await request.json()) as {
          name: string;
          description?: string;
          permissions?: APIKeyPermission[];
        };

        const createdKey: ApiKeyRecord = {
          id: `key-${apiKeys.length + 1}`,
          name: body.name,
          head: "head",
          tail: "tail",
          status: APIKeyStatus.ACTIVE,
        };

        apiKeys = [...apiKeys, createdKey];

        return {
          api_key: toApiKeyResponse(createdKey),
          plain_text_key: "plain-text-key",
        };
      }),
      getDeleteV1RevokeApiKeyMockHandler(({ params }) => {
        const keyId = String(params.keyId);
        const removedKey = apiKeys.find((key) => key.id === keyId);

        revokedKeyId = keyId;
        apiKeys = apiKeys.filter((key) => key.id !== keyId);

        return toApiKeyResponse(
          removedKey ?? {
            id: keyId,
            name: "Unknown key",
            head: "head",
            tail: "tail",
            status: APIKeyStatus.REVOKED,
          },
        );
      }),
    );
  });

  test("creates a new API key", async () => {
    render(<ApiKeysPage />);

    fireEvent.click(await screen.findByText("Create Key"));
    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "CLI Key" },
    });
    fireEvent.click(screen.getByText("Create"));

    expect(
      await screen.findByText("AutoGPT Platform API Key Created"),
    ).toBeDefined();

    await waitFor(() => {
      expect(apiKeys[0]?.name).toBe("CLI Key");
    });
  });

  test("revokes an existing API key", async () => {
    apiKeys = [
      {
        id: "key-1",
        name: "Existing Key",
        head: "head",
        tail: "tail",
        status: APIKeyStatus.ACTIVE,
      },
    ];

    render(<ApiKeysPage />);

    expect(await screen.findByText("Existing Key")).toBeDefined();

    fireEvent.pointerDown(screen.getByTestId("api-key-actions"));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Revoke" }));

    await waitFor(() => {
      expect(revokedKeyId).toBe("key-1");
    });
  });
});
