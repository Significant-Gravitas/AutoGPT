import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import type { AirtableBaseAccess } from "@/app/api/__generated__/models/airtableBaseAccess";
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
  resource_picker: {
    provider: "airtable",
    access: "create",
    credentials_id: "cred-1",
  },
};

// What run_capability returns when a connected Airtable step has no base yet.
function pickerCard(picker: Record<string, unknown> = PICKER) {
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
        inputs: [picker],
        execution_modes: ["immediate"],
      },
    },
    graph_id: null,
    graph_version: null,
  } as unknown as SetupRequirementsResponse;
}

function serveBases(
  bases: { id: string; name: string; access: AirtableBaseAccess }[],
  onCall?: () => void,
) {
  server.use(
    http.get(BASES_URL, () => {
      onCall?.();
      return HttpResponse.json({ bases, truncated: false });
    }),
  );
}

async function openPicker() {
  const trigger = await screen.findByRole("combobox", {
    name: "Choose an Airtable base",
  });
  fireEvent.click(trigger);
  await screen.findByRole("listbox");
  return trigger;
}

afterEach(() => {
  mockOnSend.mockReset();
});

describe("Airtable base picker in the setup card", () => {
  it("lists the account's bases by name and sends the one picked", async () => {
    serveBases([
      { id: "app2", name: "Shop orders", access: "create" },
      { id: "app3", name: "Craft fair", access: "create" },
      { id: "app1", name: "Wholesale prices", access: "read" },
    ]);
    render(<SetupRequirementsCard output={pickerCard()} />);

    const trigger = await screen.findByRole("combobox", {
      name: "Choose an Airtable base",
    });
    expect(trigger.textContent).toBe("Choose a base");
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(
      screen.getByText(/A base is Airtable's word for a set of tables/),
    ).toBeDefined();
    expect(screen.queryByText(/workspace/i)).toBeNull();
    const proceed = screen.getByRole("button", { name: "Proceed" });
    expect(proceed.hasAttribute("disabled")).toBe(true);

    await openPicker();
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    const readOnly = screen.getByRole("option", { name: /Wholesale prices/ });
    expect(readOnly.hasAttribute("disabled")).toBe(true);
    expect(screen.getByText("View only")).toBeDefined();
    expect(
      screen.queryByRole("textbox", { name: "Search your bases" }),
    ).toBeNull();

    fireEvent.click(screen.getByRole("option", { name: "Shop orders" }));
    await waitFor(() => expect(screen.queryByRole("listbox")).toBeNull());
    expect(trigger.textContent).toBe("Shop orders");
    await waitFor(() => expect(proceed.hasAttribute("disabled")).toBe(false));
    fireEvent.click(proceed);

    await waitFor(() => expect(mockOnSend).toHaveBeenCalledTimes(1));
    expect(mockOnSend.mock.calls[0][0]).toContain('"base_id": "app2"');
  });

  it("the picked base is marked when the list opens again", async () => {
    serveBases([
      { id: "app2", name: "Shop orders", access: "create" },
      { id: "app3", name: "Craft fair", access: "create" },
    ]);
    render(<SetupRequirementsCard output={pickerCard()} />);

    await openPicker();
    fireEvent.click(screen.getByRole("option", { name: "Craft fair" }));
    await waitFor(() => expect(screen.queryByRole("listbox")).toBeNull());

    await openPicker();
    expect(
      screen
        .getByRole("option", { name: "Craft fair" })
        .getAttribute("aria-selected"),
    ).toBe("true");
    expect(
      screen
        .getByRole("option", { name: "Shop orders" })
        .getAttribute("aria-selected"),
    ).toBe("false");
  });

  it("a long list can be searched, and the search clears when it closes", async () => {
    const names = [
      "Craft fair 2025",
      "Customer emails",
      "Etsy shop orders",
      "Holiday market",
      "Instagram calendar",
      "Printify products",
      "Supplier contacts",
      "Family budget",
    ];
    serveBases(
      names.map((name, index) => ({
        id: `app${index}`,
        name,
        access: "create" as const,
      })),
    );
    render(<SetupRequirementsCard output={pickerCard()} />);

    const trigger = await openPicker();
    const search = screen.getByRole("textbox", { name: "Search your bases" });
    fireEvent.change(search, { target: { value: "ETSY" } });
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Etsy shop orders",
    ]);

    fireEvent.change(search, { target: { value: "nothing like it" } });
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    expect(screen.getByText(/No base matches “nothing like it”/)).toBeDefined();

    fireEvent.change(search, { target: { value: "etsy" } });
    fireEvent.keyDown(search, { key: "Enter" });
    await waitFor(() => expect(screen.queryByRole("listbox")).toBeNull());
    expect(trigger.textContent).toBe("Etsy shop orders");

    await openPicker();
    expect(
      (
        screen.getByRole("textbox", {
          name: "Search your bases",
        }) as HTMLInputElement
      ).value,
    ).toBe("");
    expect(screen.getAllByRole("option")).toHaveLength(names.length);
  });

  it.each([
    ["create", /you'll see what it adds before anything changes/],
    ["edit", /Otto changes records in the one you pick/],
    ["read", /Otto only reads from the one you pick/],
  ] as const)(
    "the hint says what a %s step does to the base, and no more",
    async (access, words) => {
      serveBases([{ id: "app2", name: "Shop orders", access: "create" }]);
      render(
        <SetupRequirementsCard
          output={pickerCard({
            ...PICKER,
            resource_picker: {
              provider: "airtable",
              access,
              credentials_id: "cred-1",
            },
          })}
        />,
      );

      expect(await screen.findByText(words)).toBeDefined();
      if (access !== "create") {
        expect(screen.queryByText(/before anything changes/)).toBeNull();
      }
    },
  );

  it("a step that only reads can use a view-only base", async () => {
    serveBases([
      { id: "app2", name: "Shop orders", access: "create" },
      { id: "app1", name: "Wholesale prices", access: "read" },
    ]);
    render(
      <SetupRequirementsCard
        output={pickerCard({
          ...PICKER,
          resource_picker: {
            provider: "airtable",
            access: "read",
            credentials_id: "cred-1",
          },
        })}
      />,
    );

    const trigger = await openPicker();
    const wholesale = screen.getByRole("option", { name: "Wholesale prices" });
    expect(wholesale.hasAttribute("disabled")).toBe(false);
    expect(screen.queryByText("View only")).toBeNull();
    fireEvent.click(wholesale);
    await waitFor(() => expect(trigger.textContent).toBe("Wholesale prices"));
  });

  it("before a record change, view-only bases say records can't change", async () => {
    serveBases([{ id: "app1", name: "Wholesale prices", access: "read" }]);
    render(
      <SetupRequirementsCard
        output={pickerCard({
          ...PICKER,
          resource_picker: {
            provider: "airtable",
            access: "edit",
            credentials_id: "cred-1",
          },
        })}
      />,
    );

    expect(
      await screen.findByText(/so Otto can't change records in them/),
    ).toBeDefined();
    expect(screen.queryByText(/add/)).toBeNull();
  });

  it("a step that adds tables can't use a base the user may only edit", async () => {
    serveBases([
      { id: "app2", name: "Shop orders", access: "create" },
      { id: "app3", name: "Team tracker", access: "edit" },
    ]);
    render(<SetupRequirementsCard output={pickerCard()} />);

    const trigger = await openPicker();
    const tracker = screen.getByRole("option", { name: /Team tracker/ });
    expect(tracker.hasAttribute("disabled")).toBe(true);
    expect(screen.getByText("Needs creator access")).toBeDefined();
    fireEvent.click(tracker);
    expect(trigger.textContent).toBe("Choose a base");
  });

  it("arrow keys move past a view-only base", async () => {
    serveBases([
      { id: "app2", name: "Shop orders", access: "create" },
      { id: "app1", name: "Wholesale prices", access: "read" },
      { id: "app3", name: "Craft fair", access: "create" },
    ]);
    render(<SetupRequirementsCard output={pickerCard()} />);

    await openPicker();
    const shop = screen.getByRole("option", { name: "Shop orders" });
    fireEvent.keyDown(shop, { key: "ArrowDown" });
    expect(document.activeElement).toBe(
      screen.getByRole("option", { name: "Craft fair" }),
    );
    fireEvent.keyDown(document.activeElement!, { key: "ArrowDown" });
    expect(document.activeElement).toBe(shop);
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

  it("a busy Airtable says to wait, not to reconnect", async () => {
    server.use(
      http.get(BASES_URL, () =>
        HttpResponse.json(
          { detail: "Airtable is busy right now. Try again in a minute." },
          { status: 503 },
        ),
      ),
    );
    render(<SetupRequirementsCard output={pickerCard()} />);

    expect(await screen.findByText(/Airtable is busy right now/)).toBeDefined();
    expect(screen.queryByText(/Reconnect/)).toBeNull();
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
