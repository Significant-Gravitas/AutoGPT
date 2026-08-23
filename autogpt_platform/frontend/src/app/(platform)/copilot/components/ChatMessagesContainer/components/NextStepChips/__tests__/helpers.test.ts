import { describe, expect, it } from "vitest";
import type { MessagePart } from "../../../helpers";
import { getNextStepSuggestions } from "../helpers";

function suggestionsPart(suggestions: unknown): MessagePart {
  return {
    type: "data-suggestions",
    data: { suggestions },
  } as unknown as MessagePart;
}

const TEXT_PART = { type: "text", text: "Report is ready." } as MessagePart;

describe("getNextStepSuggestions", () => {
  it("returns the labels from a data-suggestions part", () => {
    expect(
      getNextStepSuggestions([
        TEXT_PART,
        suggestionsPart(["Email the report", "Post on r/SaaS"]),
      ]),
    ).toEqual(["Email the report", "Post on r/SaaS"]);
  });

  it("returns nothing when the message carries no suggestions part", () => {
    expect(getNextStepSuggestions([TEXT_PART])).toEqual([]);
  });

  it("keeps only the last part so a replayed stream does not stack rows", () => {
    expect(
      getNextStepSuggestions([
        suggestionsPart(["Stale suggestion"]),
        suggestionsPart(["Email the report"]),
      ]),
    ).toEqual(["Email the report"]);
  });

  it("caps the row at three chips", () => {
    expect(
      getNextStepSuggestions([suggestionsPart(["a", "b", "c", "d", "e"])]),
    ).toEqual(["a", "b", "c"]);
  });

  it("drops blanks and non-strings instead of rendering empty chips", () => {
    expect(
      getNextStepSuggestions([suggestionsPart(["  Email  ", "", 7, null])]),
    ).toEqual(["Email"]);
  });

  it("ignores a malformed payload rather than throwing", () => {
    expect(getNextStepSuggestions([suggestionsPart("not an array")])).toEqual(
      [],
    );
    expect(
      getNextStepSuggestions([
        { type: "data-suggestions" } as unknown as MessagePart,
      ]),
    ).toEqual([]);
  });
});
