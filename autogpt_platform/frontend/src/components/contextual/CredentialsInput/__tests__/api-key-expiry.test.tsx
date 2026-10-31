import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import useCredentials from "@/hooks/useCredentials";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";

import { APIKeyCredentialsModal } from "../components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { CredentialTypeSelector } from "../components/CredentialTypeSelector/CredentialTypeSelector";

// #16234: the builder's API key forms keep the picked expiry time and refuse
// one that has already passed.

vi.mock("@/hooks/useCredentials", () => ({ default: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn(), toasts: [] }),
}));

const mockUseCredentials = useCredentials as unknown as ReturnType<
  typeof vi.fn
>;

const schema = {
  type: "object",
  properties: {},
  credentials_provider: ["openai"],
  credentials_types: ["api_key"],
} as unknown as BlockIOCredentialsSubSchema;

const createAPIKeyCredentials = vi.fn();
const originalTZ = process.env.TZ;

beforeEach(() => {
  // Chicago, 2026-10-26 10:30 CDT (UTC-5).
  process.env.TZ = "America/Chicago";
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date("2026-10-26T15:30:00Z"));
  createAPIKeyCredentials.mockResolvedValue({ id: "new", title: "Key" });
  mockUseCredentials.mockReturnValue({
    isLoading: false,
    provider: "openai",
    providerName: "OpenAI",
    supportsApiKey: true,
    supportsUserPassword: false,
    supportsHostScoped: false,
    discriminatorValue: undefined,
    savedCredentials: [],
    createAPIKeyCredentials,
  });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.clearAllMocks();
  process.env.TZ = originalTZ;
});

function fill(expiresAt: string) {
  const type = (placeholder: string, value: string) =>
    fireEvent.change(screen.getByPlaceholderText(placeholder), {
      target: { value },
    });
  type("Enter a name for this API Key...", "My key");
  type("Enter API Key...", "sk-abc");
  type("Select expiration date...", expiresAt);
  fireEvent.click(screen.getByRole("button", { name: "Add API Key" }));
}

function renderModal() {
  render(
    <APIKeyCredentialsModal
      schema={schema}
      open
      onClose={vi.fn()}
      onCredentialsCreate={vi.fn()}
    />,
  );
}

function renderSelector() {
  render(
    <CredentialTypeSelector
      schema={schema}
      open
      onClose={vi.fn()}
      provider="openai"
      providerName="OpenAI"
      supportedTypes={["api_key"]}
      onCredentialsCreate={vi.fn()}
      onOAuthLogin={vi.fn()}
      onOpenPasswordModal={vi.fn()}
      onOpenHostScopedModal={vi.fn()}
    />,
  );
}

describe("APIKeyCredentialsModal expiry", () => {
  it("keeps the picked time later today instead of midnight", async () => {
    renderModal();
    fill("2026-10-26T20:15");

    await waitFor(() =>
      expect(createAPIKeyCredentials).toHaveBeenCalledWith(
        // 2026-10-26T20:15:00-05:00
        expect.objectContaining({ expires_at: 1793063700 }),
      ),
    );
    expect(
      (
        screen.getByPlaceholderText(
          "Select expiration date...",
        ) as HTMLInputElement
      ).value,
    ).toBe("2026-10-26T20:15");
  });

  it("rejects a time earlier today without saving", async () => {
    renderModal();
    fill("2026-10-26T04:00");

    expect(
      await screen.findByText("Expiry must be in the future"),
    ).toBeDefined();
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  it("rejects yesterday without saving", async () => {
    renderModal();
    fill("2026-10-25T10:00");

    expect(
      await screen.findByText("Expiry must be in the future"),
    ).toBeDefined();
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  it("sets min on the picker to now", () => {
    renderModal();

    expect(
      screen
        .getByPlaceholderText("Select expiration date...")
        .getAttribute("min"),
    ).toBe("2026-10-26T10:30");
  });
});

describe("CredentialTypeSelector API key tab expiry", () => {
  it("rejects a past expiry without saving", async () => {
    renderSelector();
    fill("2026-10-25T10:00");

    expect(
      await screen.findByText("Expiry must be in the future"),
    ).toBeDefined();
    expect(createAPIKeyCredentials).not.toHaveBeenCalled();
  });

  it("saves a future expiry at the picked time", async () => {
    renderSelector();
    fill("2026-10-26T20:15");

    await waitFor(() =>
      expect(createAPIKeyCredentials).toHaveBeenCalledWith(
        expect.objectContaining({ expires_at: 1793063700 }),
      ),
    );
  });
});
