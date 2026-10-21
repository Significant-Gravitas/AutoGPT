import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { getVisibleUserMessageParts } from "../userMessageParts";

interface PreambleCase {
  kind: string;
  caller: string;
  context: string;
  prompt: string;
  message: string;
  visible: string;
}

// The backend's copy on purpose: the preamble wording lives in
// delegate_to_expert.py / handoff_to_expert.py, and this one file keeps the
// frontend matcher in step with it.
const preambleCases: { cases: PreambleCase[] } = JSON.parse(
  readFileSync(
    resolve(
      dirname(fileURLToPath(import.meta.url)),
      "../../../../../../../../backend/backend/copilot/tools/delegation_preamble_cases.json",
    ),
    "utf8",
  ),
);

describe("the shared backend/frontend delegation preamble", () => {
  it.each(preambleCases.cases)(
    "hides the $kind preamble the backend renders for $caller",
    (preambleCase) => {
      expect(
        getVisibleUserMessageParts([
          { type: "text", text: preambleCase.message },
        ]),
      ).toEqual([{ type: "text", text: preambleCase.visible }]);
    },
  );
});
