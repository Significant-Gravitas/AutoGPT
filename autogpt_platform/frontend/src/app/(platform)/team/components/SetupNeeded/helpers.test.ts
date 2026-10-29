import type { ExpertSetupItem } from "@/app/api/__generated__/models/expertSetupItem";
import { describe, expect, test } from "vitest";
import { getSetupRowCopy } from "./helpers";

function inputsItem(missing_inputs: string[]): ExpertSetupItem {
  return {
    expert_id: "expert-frankie",
    expert_name: "Frankie",
    expert_avatar_url: null,
    workflow_id: "wf-1",
    workflow_name: "Personal Newsletter",
    library_agent_id: "lib-1",
    providers: [],
    resolution: "inputs",
    missing_inputs,
  };
}

describe("getSetupRowCopy for missing inputs", () => {
  test.each([
    [[], "Needs a few details from you before it can run on schedule."],
    [["Email Address"], "Needs Email Address before it can run on schedule."],
    [
      ["Email Address", "Topic"],
      "Needs Email Address and Topic before it can run on schedule.",
    ],
    [
      ["Email Address", "Topic", "Send Time"],
      "Needs Email Address, Topic and Send Time before it can run on schedule.",
    ],
  ])("%j", (missing, detail) => {
    expect(getSetupRowCopy(inputsItem(missing)).detail).toBe(detail);
  });
});
