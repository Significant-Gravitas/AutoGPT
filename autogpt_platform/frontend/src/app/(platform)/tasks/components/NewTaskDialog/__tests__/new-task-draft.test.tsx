import {
  makeExpert,
  makeTask,
} from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NewTaskDialog } from "../NewTaskDialog";

const EXPERTS = [makeExpert("maria", "Maria")];

function Host({ experts }: { experts: Expert[] }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setIsOpen(true)}>
        Open
      </button>
      <NewTaskDialog
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        experts={experts}
      />
    </>
  );
}

function field() {
  return screen.getByLabelText("What needs doing?") as HTMLTextAreaElement;
}

async function open(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Open" }));
  await screen.findByRole("dialog", { name: "New task" });
}

async function closeWithEscape(user: ReturnType<typeof userEvent.setup>) {
  await user.keyboard("{Escape}");
  await vi.waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "New task" })).toBeNull(),
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
  useAuthStore.setState({ user: { id: "user-1" } as User });
  server.use(
    http.post("/api/proxy/api/tasks", () =>
      HttpResponse.json(makeTask("task-9", "Call the supplier", "TODO"), {
        status: 201,
      }),
    ),
    http.get("/api/proxy/api/tasks", () => HttpResponse.json({ tasks: [] })),
  );
});

afterEach(() => {
  useAuthStore.setState({ user: null });
});

describe("New task draft", () => {
  it("keeps the text, owner and options after leaving the page and coming back", async () => {
    const user = userEvent.setup();
    const first = render(<Host experts={EXPERTS} />);
    await open(user);
    await user.type(
      field(),
      "Call the supplier{Shift>}{Enter}{/Shift}About the invoices",
    );
    await user.click(
      screen.getByRole("button", { name: "Owner: Otto. Change owner" }),
    );
    await user.click(
      await screen.findByRole("menuitemradio", { name: "Maria" }),
    );
    await user.click(screen.getByRole("button", { name: "Options" }));
    await user.type(screen.getByLabelText("Done when"), "Order confirmed");
    first.unmount();

    render(<Host experts={EXPERTS} />);
    await open(user);

    expect(field().value).toBe("Call the supplier\nAbout the invoices");
    expect(
      screen.getByRole("button", { name: "Owner: Maria. Change owner" }),
    ).toBeDefined();
    expect((screen.getByLabelText("Done when") as HTMLInputElement).value).toBe(
      "Order confirmed",
    );
  });

  it("keeps the draft when the dialog is closed, and forgets it on Discard", async () => {
    const user = userEvent.setup();
    render(<Host experts={EXPERTS} />);
    await open(user);
    await user.type(field(), "Call the supplier");
    await closeWithEscape(user);

    await open(user);
    expect(field().value).toBe("Call the supplier");

    await user.click(screen.getByRole("button", { name: "Discard" }));
    await vi.waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "New task" })).toBeNull(),
    );
    await open(user);
    expect(field().value).toBe("");
  });

  it("forgets the draft once the task is created", async () => {
    const user = userEvent.setup();
    render(<Host experts={EXPERTS} />);
    await open(user);
    await user.type(field(), "Call the supplier");
    await user.click(screen.getByRole("button", { name: "Save for later" }));
    await vi.waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "New task" })).toBeNull(),
    );

    await open(user);
    expect(field().value).toBe("");
  });

  it("keeps another person's draft to themselves", async () => {
    const user = userEvent.setup();
    const first = render(<Host experts={EXPERTS} />);
    await open(user);
    await user.type(field(), "Private plan");
    first.unmount();

    useAuthStore.setState({ user: { id: "user-2" } as User });
    render(<Host experts={EXPERTS} />);
    await open(user);

    expect(field().value).toBe("");
  });
});
