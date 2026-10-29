import { describe, expect, it } from "vitest";
import type { getV1ListAvailableBlocksResponse } from "@/app/api/__generated__/endpoints/blocks/blocks";
import { getAnimationText, getBlockNamesById } from "../helpers";

function make200(
  blocks: Record<string, unknown>[],
): getV1ListAvailableBlocksResponse {
  return {
    data: blocks,
    status: 200,
    headers: new Headers(),
  };
}

describe("getBlockNamesById", () => {
  it("returns undefined without a response", () => {
    expect(getBlockNamesById(undefined)).toBeUndefined();
  });

  it("returns undefined for a non-200 response", () => {
    const response = {
      data: { detail: "Not authenticated" },
      status: 401,
      headers: new Headers(),
    } as getV1ListAvailableBlocksResponse;
    expect(getBlockNamesById(response)).toBeUndefined();
  });

  it("maps ids to beautified names without the Block suffix", () => {
    const names = getBlockNamesById(
      make200([{ id: "block-1", name: "AIConversationBlock" }]),
    );
    expect(names?.get("block-1")).toBe("AI Conversation");
  });

  it("skips entries without string id and name", () => {
    const names = getBlockNamesById(
      make200([
        { id: "block-1" },
        { name: "OrphanBlock" },
        { id: 42, name: "NumericIdBlock" },
        { id: "block-2", name: "GetWeatherBlock" },
      ]),
    );
    expect(names?.size).toBe(1);
    expect(names?.get("block-2")).toBe("Get Weather");
  });
});

describe("getAnimationText block name resolution", () => {
  const blockId = "87840993-2053-44b7-8da4-187ad4ee518c";

  it("prefers block_name from the input", () => {
    const text = getAnimationText(
      {
        state: "input-available",
        input: { block_id: blockId, block_name: "My Block" },
      },
      new Map([[blockId, "Looked Up"]]),
    );
    expect(text).toBe('Running "My Block"');
  });

  it("falls back to the looked-up name when input has only block_id", () => {
    const text = getAnimationText(
      { state: "input-available", input: { block_id: blockId } },
      new Map([[blockId, "AI Conversation"]]),
    );
    expect(text).toBe('Running "AI Conversation"');
  });

  it("falls back to the raw block_id when no name is known", () => {
    expect(
      getAnimationText({
        state: "input-available",
        input: { block_id: blockId },
      }),
    ).toBe(`Running "${blockId}"`);
    expect(
      getAnimationText(
        { state: "input-available", input: { block_id: blockId } },
        new Map([["other-id", "Other"]]),
      ),
    ).toBe(`Running "${blockId}"`);
  });

  it("uses the looked-up name for dry runs too", () => {
    const text = getAnimationText(
      {
        state: "input-available",
        input: { block_id: blockId, dry_run: true },
      },
      new Map([[blockId, "AI Conversation"]]),
    );
    expect(text).toBe('Simulating "AI Conversation"');
  });
});
