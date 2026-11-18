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
    const userId = "user-1";
    const earlier = withScheduleCreateKey({ userId, create });
    forgetScheduleCreateKey({ userId, create: earlier });
    const newer = withScheduleCreateKey({ userId, create });

    forgetScheduleCreateKey({ userId, create: earlier });

    expect(withScheduleCreateKey({ userId, create }).data.schedule_id).toBe(
      newer.data.schedule_id,
    );
    expect(newer.data.schedule_id).not.toBe(earlier.data.schedule_id);
  });

  test("another account signed in to the same tab doesn't inherit a key", () => {
    const first = withScheduleCreateKey({ userId: "user-1", create });

    const other = withScheduleCreateKey({ userId: "user-2", create });

    expect(other.data.schedule_id).not.toBe(first.data.schedule_id);
    expect(
      withScheduleCreateKey({ userId: "user-1", create }).data.schedule_id,
    ).toBe(first.data.schedule_id);
  });
});
