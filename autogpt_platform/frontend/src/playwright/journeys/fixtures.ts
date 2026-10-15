import { PW_FLAG_OVERRIDES_STORAGE_KEY } from "../../services/feature-flags/pw-flag-overrides";
import { expect, test as base } from "../coverage-fixture";
import { type ExpectedResponse, watchForUnexpectedErrors } from "./guard";
import hostedFlags from "./hosted-flags.json";
import { type JourneyUser, provisionJourneyUser } from "./journey-user";

interface JourneyOptions {
  /** API responses of 400 or more this journey provokes on purpose. */
  expectedResponses: ExpectedResponse[];
  /** Flag values to change from the hosted defaults; the rest are kept. */
  flagOverrides: Record<string, unknown>;
}

interface JourneyFixtures {
  journeyGuard: void;
  /** A fresh, signed-in user who has finished onboarding. */
  journeyUser: JourneyUser;
}

export const test = base.extend<JourneyOptions & JourneyFixtures>({
  expectedResponses: [[], { option: true }],
  flagOverrides: [{}, { option: true }],

  journeyGuard: [
    async ({ context, expectedResponses, flagOverrides }, use) => {
      await context.addInitScript(
        ({ key, value }) => {
          try {
            window.localStorage.setItem(key, value);
          } catch {
            // Opaque origins (about:blank, data:) have no storage.
          }
        },
        {
          key: PW_FLAG_OVERRIDES_STORAGE_KEY,
          value: JSON.stringify({ ...hostedFlags, ...flagOverrides }),
        },
      );
      const guard = watchForUnexpectedErrors(context, expectedResponses);

      await use();

      const problems = await guard.collect();
      expect(
        problems,
        `the journey hit ${problems.length} unexpected error(s). Declare ` +
          "deliberate ones with test.use({ expectedResponses }):\n\n" +
          problems.join("\n\n"),
      ).toEqual([]);
    },
    { auto: true },
  ],

  journeyUser: async ({ page }, provide) => {
    await provide(await provisionJourneyUser(page));
  },
});

export { expect };
