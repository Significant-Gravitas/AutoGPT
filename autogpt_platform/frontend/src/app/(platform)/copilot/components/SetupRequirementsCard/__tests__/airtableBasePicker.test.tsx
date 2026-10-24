import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { isAirtableBasePickerSchema } from "@/components/renderers/InputRenderer/custom/AirtableBasePickerField/helpers";
import { SetupRequirementsCard } from "../SetupRequirementsCard";

const mockOnSend = vi.fn();
vi.mock("../../CopilotChatActionsProvider/useCopilotChatActions", () => ({
  useCopilotChatActions: () => ({ onSend: mockOnSend }),
}));

const BASES_URL = "*/api/integrations/airtable/credentials/cred-1/bases";

const PICKER = {
  name: "base_id",
  title: "Airtable base",
  type: "string",
  required: true,
  advanced: false,
  description: "The Airtable base to use.",
  format: "airtable-base-picker",
  resource_picker: { provider: "airtable", credentials_id: "cred-1" },
};

// What run_capability returns when a connected Airtable step has no base yet.
function pickerCard(): SetupRequirementsResponse {
  return {
    type: "setup_requirements",
    message:
      "Choose your Airtable base in the card below. The step runs again with your choice.",
    session_id: "s1",
    setup_info: {
      agent_id: "fcc20ced-d817-42ea-9b40-c35e7bf34b4f",
      agent_name: "AirtableCreateTableBlock",
      user_readiness: {
        has_all_credentials: true,
        missing_credentials: {},
        ready_to_run: false,
      },
      requirements: {
        credentials: [],
        inputs: [PICKER],
        execution_modes: ["immediate"],
      },
    },
    graph_id: null,
    graph_version: null,
  } as unknown as SetupRequirementsResponse;
}

function serveBases(
  bases: { id: string; name: string; can_edit: boolean }[],
  onCall?: () => void,
) {
  server.use(
    http.get(BASES_URL, () => {
      onCall?.();
      return HttpResponse.json({ bases, truncated: false });
    }),
  );
}

afterEach(() => {
  mockOnSend.mockReset();
});

describe("Airtable base picker in the setup card", () => {
  it("lists the account's bases by name and sends the one picked", async () => {
    serveBases([
      { id: "app2", name: "Shop orders", can_edit: true },
      { id: "app3", name: "Craft fair", can_edit: true },
      { id: "app1", name: "Wholesale prices", can_edit: false },
    ]);
    render(<SetupRequirementsCard output={pickerCard()} />);

    const shop = await screen.findByRole("radio", { name: "Shop orders" });
    const readOnly = screen.getByRole("radio", { name: /Wholesale prices/ });
    expect(readOnly.hasAttribute("disabled")).toBe(true);
    expect(screen.getByText("View only")).toBeDefined();
    expect(
      screen.getByText(/A base is Airtable's word for a set of tables/),
    ).toBeDefined();
    expect(screen.queryByText(/workspace/i)).toBeNull();

    const proceed = screen.getByRole("button", { name: "Proceed" });
    expect(proceed.hasAttribute("disabled")).toBe(true);
    fireEvent.click(shop);
    expect(shop.getAttribute("aria-checked")).toBe("true");
    await waitFor(() => expect(proceed.hasAttribute("disabled")).toBe(false));
    fireEvent.click(proceed);

    await waitFor(() => expect(mockOnSend).toHaveBeenCalledTimes(1));
    expect(mockOnSend.mock.calls[0][0]).toContain('"base_id": "app2"');
  });

  it("arrow keys move the pick past a view-only base", async () => {
    serveBases([
      { id: "app2", name: "Shop orders", can_edit: true },
      { id: "app1", name: "Wholesale prices", can_edit: false },
      { id: "app3", name: "Craft fair", can_edit: true },
    ]);
    render(<SetupRequirementsCard output={pickerCard()} />);

    const shop = await screen.findByRole("radio", { name: "Shop orders" });
    fireEvent.keyDown(shop, { key: "ArrowDown" });

    expect(
      screen
        .getByRole("radio", { name: "Craft fair" })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  it("with no bases yet, points to Airtable and lists again on refresh", async () => {
    let calls = 0;
    serveBases([], () => {
      calls += 1;
    });
    render(<SetupRequirementsCard output={pickerCard()} />);

    expect(
      await screen.findByText(/Your Airtable has no bases yet/),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: /Open Airtable/ }).getAttribute("href"),
    ).toBe("https://airtable.com/");
    fireEvent.click(screen.getByRole("button", { name: "Refresh list" }));
    await waitFor(() => expect(calls).toBe(2));
  });

  it("a refused connection says how to fix it", async () => {
    server.use(
      http.get(BASES_URL, () =>
        HttpResponse.json(
          {
            detail:
              "Airtable didn't accept this connection. Reconnect Airtable and try again.",
          },
          { status: 409 },
        ),
      ),
    );
    render(<SetupRequirementsCard output={pickerCard()} />);

    expect(
      await screen.findByText(/Reconnect Airtable and try again/),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
  });

  it("without the account the run uses, the field stays a plain input", () => {
    expect(isAirtableBasePickerSchema(PICKER)).toBe(true);
    expect(
      isAirtableBasePickerSchema({
        ...PICKER,
        resource_picker: { provider: "airtable" },
      }),
    ).toBe(false);
  });
});
