import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test } from "vitest";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import {
  folder,
  mail,
} from "@/app/(platform)/copilot/components/ApprovalQueue/__tests__/fixtures";
import {
  ada,
  homeHeldItem,
  leo,
  makeDashboard,
} from "@/app/(platform)/home/__tests__/heldItems";
import { ExpertNeedsYouSection } from "../ExpertNeedsYouSection";

test("an Expert's page reviews its held calls with Home's row and dialog", async () => {
  const user = userEvent.setup();
  const email = homeHeldItem(mail("m1"), { expert: ada });
  const bare = homeHeldItem(folder("f1", "Q3 reports"), { expert: ada });
  const elsewhere = homeHeldItem(folder("f2", "Invoices"), { expert: leo });
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([email, bare, elsewhere])),
    ),
  );

  render(<ExpertNeedsYouSection expert={{ id: ada.id } as Expert} enabled />);

  expect(
    await screen.findByRole("button", { name: `Review: ${email.title}` }),
  ).toBeDefined();
  expect(
    screen.queryByRole("button", { name: `Approve: ${email.title}` }),
  ).toBeNull();
  expect(
    screen.getByRole("button", { name: `Approve: ${bare.title}` }),
  ).toBeDefined();
  expect(screen.queryByText("Invoices")).toBeNull();

  await user.click(
    screen.getByRole("button", { name: `Review: ${email.title}` }),
  );
  const dialog = await screen.findByRole("dialog");
  expect(dialog.textContent).toContain("dana@acme.com");
  // Scoped to this Expert: two held calls, never Leo's.
  expect(within(dialog).getByText("1 of 2")).toBeDefined();
});
