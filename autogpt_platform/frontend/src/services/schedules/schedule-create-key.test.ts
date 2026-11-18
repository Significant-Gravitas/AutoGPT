import { afterEach, describe, expect, test } from "vitest";
import {
  forgetScheduleCreateKey,
  withScheduleCreateKey,
} from "./schedule-create-key";

const create = {
  graphId: "graph-1",
  data: { name: "Morning run", cron: "0 9 * * *", inputs: { topic: "news" } },
};

afterEach(() => {
  window.sessionStorage.clear();
});

describe("schedule create keys", () => {
  test("a late answer to an earlier attempt leaves the key a newer attempt took", () => {
    const earlier = withScheduleCreateKey(create);
    forgetScheduleCreateKey(earlier);
    const newer = withScheduleCreateKey(create);

    forgetScheduleCreateKey(earlier);

    expect(withScheduleCreateKey(create).data.schedule_id).toBe(
      newer.data.schedule_id,
    );
    expect(newer.data.schedule_id).not.toBe(earlier.data.schedule_id);
  });
});
