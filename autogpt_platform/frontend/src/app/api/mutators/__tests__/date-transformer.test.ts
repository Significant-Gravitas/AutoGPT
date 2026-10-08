import { describe, expect, it } from "vitest";

import { transformDates } from "../date-transformer";

describe("transformDates", () => {
  it("converts server timestamps with an explicit Z to Date", () => {
    const out = transformDates({ created_at: "2024-03-10T09:30:00Z" });
    expect(out.created_at).toBeInstanceOf(Date);
    expect((out.created_at as unknown as Date).toISOString()).toBe(
      "2024-03-10T09:30:00.000Z",
    );
  });

  it("leaves zone-less timestamps as strings", () => {
    const out = transformDates({ started_at: "2024-03-10T09:30:00" });
    expect(out.started_at).toBe("2024-03-10T09:30:00");
  });

  it("does not touch user graph inputs/outputs", () => {
    const payload = {
      id: "exec-1",
      created_at: "2024-03-10T09:30:00Z",
      inputs: { when: "2024-03-10T09:30:00", at: "2024-03-10T09:30:00Z" },
      outputs: { t: ["2024-03-10T09:30:00Z"] },
      input_data: { when: "2024-03-10T09:30:00Z" },
      output_data: { t: ["2024-03-10T09:30:00Z"] },
      credential_inputs: { c: { expires: "2024-03-10T09:30:00Z" } },
      nodes_input_masks: { n1: { when: "2024-03-10T09:30:00Z" } },
    };
    const out = transformDates(payload);

    expect(out.created_at).toBeInstanceOf(Date);
    for (const key of [
      "inputs",
      "outputs",
      "input_data",
      "output_data",
      "credential_inputs",
      "nodes_input_masks",
    ] as const) {
      expect(JSON.stringify(out[key])).toBe(JSON.stringify(payload[key]));
    }
  });

  it("skips user payloads nested inside lists of executions", () => {
    const out = transformDates([
      {
        started_at: "2024-03-10T09:30:00Z",
        inputs: { d: "2024-03-10T09:30:00Z" },
      },
    ]);
    expect(out[0].started_at).toBeInstanceOf(Date);
    expect(out[0].inputs.d).toBe("2024-03-10T09:30:00Z");
  });
});
