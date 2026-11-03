import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { CHAT_SESSION, folder } from "../../ApprovalQueue/__tests__/fixtures";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../CopilotPendingReviews";

test("answers on an idle chat follow no turn until the last card, which follows one", async () => {
  const chat = serveChat([folder("a", "Alpha"), folder("b", "Beta")]);
  const onBackendTurn = renderChat();
  const box = await screen.findByRole("region", { name: "Waiting for you" });

  await userEvent.click(
    within(box).getAllByRole("button", { name: "Approve" })[0],
  );
  await within(box).findByText("· Approved");
  expect(onBackendTurn).not.toHaveBeenCalled();

  await userEvent.click(within(box).getByRole("button", { name: "Approve" }));

  await waitFor(() => expect(onBackendTurn).toHaveBeenCalledTimes(1));
  expect(chat.answered).toEqual([[id("a")], [id("b")]]);
  expect(chat.waited).toEqual([true, true]);
  expect(chat.continues).toBe(0);
});

test("Close and continue starts the turn with what is answered and folds the box to its header", async () => {
  const chat = serveChat([folder("a", "Alpha"), folder("b", "Beta")]);
  const onBackendTurn = renderChat();
  const box = await screen.findByRole("region", { name: "Waiting for you" });
  expect(
    within(box).queryByRole("button", { name: "Close and continue" }),
  ).toBeNull();

  await userEvent.click(
    within(box).getAllByRole("button", { name: "Approve" })[0],
  );
  await userEvent.click(
    await within(box).findByRole("button", { name: "Close and continue" }),
  );

  await waitFor(() => expect(onBackendTurn).toHaveBeenCalledTimes(1));
  expect(chat.continues).toBe(1);
  const show = await within(box).findByRole("button", {
    name: "Show waiting cards",
  });
  await waitFor(() => expect(document.activeElement).toBe(show));
  expect(within(box).queryByRole("button", { name: "Approve" })).toBeNull();

  await userEvent.click(
    within(box).getByRole("button", { name: "Show waiting cards" }),
  );
  expect(within(box).getByRole("button", { name: "Approve" })).toBeDefined();
  expect(
    within(box).queryByRole("button", { name: "Close and continue" }),
  ).toBeNull();
});

test("an answer waiting on the chat's other cards keeps Close and continue after a reload", async () => {
  const chat = serveChat([folder("a", "Alpha"), folder("b", "Beta")]);
  const first = mountChat();
  const box = await screen.findByRole("region", { name: "Waiting for you" });
  await userEvent.click(
    within(box).getAllByRole("button", { name: "Approve" })[0],
  );
  await within(box).findByRole("button", { name: "Close and continue" });
  first.unmount();

  mountChat();
  const reloaded = await screen.findByRole("region", {
    name: "Waiting for you",
  });
  await within(reloaded).findByText("· Approved");
  await userEvent.click(
    await within(reloaded).findByRole("button", { name: "Close and continue" }),
  );

  await waitFor(() => expect(chat.continues).toBe(1));
  expect(chat.answered).toEqual([[id("a")]]);
});

test("the review dialog's last verdict follows the turn at once, and closing it sends nothing more", async () => {
  const user = userEvent.setup();
  const chat = serveChat([folder("a", "Alpha"), folder("b", "Beta")]);
  const onBackendTurn = renderChat();
  const dialog = await openReviewAll(user);

  await user.click(within(dialog).getByRole("button", { name: "Approve" }));
  await waitFor(() =>
    expect(heading(dialog)).toBe("Create library folder Beta"),
  );
  expect(onBackendTurn).not.toHaveBeenCalled();
  await user.click(within(dialog).getByRole("button", { name: "Approve" }));

  await waitFor(() => expect(onBackendTurn).toHaveBeenCalledTimes(1));
  await user.click(await within(dialog).findByRole("button", { name: "Done" }));
  expect(chat.answered).toEqual([[id("a")], [id("b")]]);
  expect(chat.continues).toBe(0);
});

test("closing the review dialog part-way continues with what it answered", async () => {
  const user = userEvent.setup();
  const chat = serveChat([folder("a", "Alpha"), folder("b", "Beta")]);
  const onBackendTurn = renderChat();
  const dialog = await openReviewAll(user);

  await user.click(within(dialog).getByRole("button", { name: "Approve" }));
  await waitFor(() =>
    expect(heading(dialog)).toBe("Create library folder Beta"),
  );
  await user.keyboard("{Escape}");

  await waitFor(() => expect(chat.continues).toBe(1));
  expect(onBackendTurn).toHaveBeenCalledTimes(1);
  expect(chat.answered).toEqual([[id("a")]]);
});

test("closing the review dialog with nothing answered sends nothing", async () => {
  const user = userEvent.setup();
  const chat = serveChat([folder("a", "Alpha"), folder("b", "Beta")]);
  const onBackendTurn = renderChat();
  await openReviewAll(user);

  await user.keyboard("{Escape}");

  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(chat.answered).toEqual([]);
  expect(chat.continues).toBe(0);
  expect(onBackendTurn).not.toHaveBeenCalled();
});

// The review endpoints as the server keeps them: an answered card leaves the
// chat's queue, and an answer says when it waits for the others.
function serveChat(reviews: PendingHumanReviewModel[]) {
  const waiting = new Map(reviews.map((r) => [r.node_exec_id, r]));
  // Answered cards whose calls have not run, until a turn runs them.
  const unrun = new Map<string, PendingHumanReviewModel>();
  const chat = {
    answered: [] as string[][],
    waited: [] as (boolean | undefined)[],
    continues: 0,
  };
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json([...waiting.values()]),
    ),
    http.post("*/api/review/action", async ({ request }) => {
      const body = (await request.json()) as {
        reviews: { node_exec_id: string; approved: boolean }[];
        wait_for_other_cards?: boolean;
      };
      const ids = body.reviews.map((r) => r.node_exec_id);
      chat.waited.push(body.wait_for_other_cards);
      body.reviews.forEach(({ node_exec_id, approved }) => {
        const row = waiting.get(node_exec_id);
        if (row)
          unrun.set(node_exec_id, {
            ...row,
            status: approved ? "APPROVED" : "REJECTED",
          });
      });
      ids.forEach((reviewId) => waiting.delete(reviewId));
      chat.answered.push(ids);
      return HttpResponse.json({
        approved_count: ids.length,
        rejected_count: 0,
        failed_count: 0,
        awaiting_other_cards: waiting.size > 0,
      });
    }),
    http.get(`*/api/review/session/${CHAT_SESSION}/answered`, () =>
      HttpResponse.json([...unrun.values()]),
    ),
    http.post(`*/api/review/session/${CHAT_SESSION}/continue`, () => {
      chat.continues += 1;
      unrun.clear();
      return new HttpResponse(null, { status: 204 });
    }),
  );
  return chat;
}

function mountChat() {
  return render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
}

function renderChat() {
  const onBackendTurn = vi.fn();
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={onBackendTurn}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
  return onBackendTurn;
}

async function openReviewAll(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: "Review all" }));
  const dialog = await screen.findByRole("dialog");
  await waitFor(() =>
    expect(heading(dialog)).toBe("Create library folder Alpha"),
  );
  return dialog;
}

function heading(dialog: HTMLElement) {
  return within(dialog)
    .getByRole("region", { name: "Held call" })
    .querySelector("h3")?.textContent;
}

function id(reviewId: string) {
  return folder(reviewId, "").node_exec_id;
}
