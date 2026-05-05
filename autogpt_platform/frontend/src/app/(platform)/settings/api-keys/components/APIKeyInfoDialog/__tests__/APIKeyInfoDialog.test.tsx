import { describe, expect, test, vi } from "vitest";

import { render, screen, within } from "@/tests/integrations/test-utils";
import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";
import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import { APIKeyStatus } from "@/app/api/__generated__/models/aPIKeyStatus";

import { APIKeyInfoDialog } from "../APIKeyInfoDialog";

function buildApiKey(overrides: Partial<APIKeyInfo> = {}): APIKeyInfo {
  return {
    id: "key_1",
    user_id: "user_1",
    name: "Production key",
    head: "pk_live_",
    tail: "abcd1234",
    status: APIKeyStatus.ACTIVE,
    scopes: [APIKeyPermission.EXECUTE_GRAPH, APIKeyPermission.READ_GRAPH],
    created_at: new Date("2026-01-15T10:30:00Z"),
    last_used_at: new Date("2026-03-20T08:15:00Z"),
    description: "Used by the production API",
    ...overrides,
  };
}

describe("APIKeyInfoDialog", () => {
  test("renders name, masked key, description, scopes, and timestamps", () => {
    const apiKey = buildApiKey();

    render(<APIKeyInfoDialog apiKey={apiKey} open onOpenChange={vi.fn()} />);

    const dialog = screen.getByRole("dialog");

    expect(within(dialog).getByText("Production key")).toBeDefined();
    expect(within(dialog).getByText(/pk_live_.+abcd1234/)).toBeDefined();
    expect(
      within(dialog).getByText(/used by the production api/i),
    ).toBeDefined();
    expect(within(dialog).getByText(/execute graph/i)).toBeDefined();
    expect(within(dialog).getByText(/read graph/i)).toBeDefined();
    expect(within(dialog).getByText(/^created$/i)).toBeDefined();
    expect(within(dialog).getByText(/^last used$/i)).toBeDefined();
  });

  test("renders 'No scopes' when the key has no permissions", () => {
    render(
      <APIKeyInfoDialog
        apiKey={buildApiKey({ scopes: [] })}
        open
        onOpenChange={vi.fn()}
      />,
    );

    expect(screen.getByText(/no scopes/i)).toBeDefined();
    expect(screen.queryByText(/execute graph/i)).toBeNull();
  });

  test("renders 'Never used' when last_used_at is missing", () => {
    render(
      <APIKeyInfoDialog
        apiKey={buildApiKey({ last_used_at: null })}
        open
        onOpenChange={vi.fn()}
      />,
    );

    expect(screen.getByText(/never used/i)).toBeDefined();
  });

  test("hides the description section when description is empty", () => {
    render(
      <APIKeyInfoDialog
        apiKey={buildApiKey({ description: "" })}
        open
        onOpenChange={vi.fn()}
      />,
    );

    expect(screen.queryByText(/^description$/i)).toBeNull();
  });

  test("does not render dialog contents when open is false", () => {
    render(
      <APIKeyInfoDialog
        apiKey={buildApiKey()}
        open={false}
        onOpenChange={vi.fn()}
      />,
    );

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
