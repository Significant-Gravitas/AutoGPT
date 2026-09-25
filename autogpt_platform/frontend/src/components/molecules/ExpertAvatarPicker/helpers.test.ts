import { ExpertAvatarRequestAccentCount } from "@/app/api/__generated__/models/expertAvatarRequestAccentCount";
import { ExpertAvatarRequestAccentPlacement } from "@/app/api/__generated__/models/expertAvatarRequestAccentPlacement";
import { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";
import { expect, test } from "vitest";
import { randomAvatarRequest } from "./helpers";

test("every rolled trait is one the backend accepts, and the category is not rolled", () => {
  const requests = Array.from({ length: 50 }, () =>
    randomAvatarRequest("sales"),
  );

  for (const request of requests) {
    expect(request.category).toBe("sales");
    expect(["standard", "light", "dark"]).toContain(request.shade);
    expect(Object.values(ExpertAvatarRequestShape)).toContain(request.shape);
    expect(Object.values(ExpertAvatarRequestBase)).toContain(request.base);
    expect(Object.values(ExpertAvatarRequestTilt)).toContain(request.tilt);
    expect(Object.values(ExpertAvatarRequestInlay)).toContain(request.inlay);
    expect(Object.values(ExpertAvatarRequestAccentPlacement)).toContain(
      request.accent_placement,
    );
    expect(Object.values(ExpertAvatarRequestAccentCount)).toContain(
      request.accent_count,
    );
    expect(Object.values(ExpertAvatarRequestExpression)).toContain(
      request.expression,
    );
  }
  expect(
    new Set(requests.map((request) => request.shape)).size,
  ).toBeGreaterThan(1);
});
