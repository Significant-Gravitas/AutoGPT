import { Locator, Page } from "@playwright/test";

export function getSelectors(page: Page) {
  return {
    getButton: makeSelector(page, getButton),
    getText: (name: string | RegExp, options?: GetByTextOptions) =>
      getText(page, name, options),
    getLink: makeSelector(page, getLink),
    getField: makeSelector(page, getField),
    getId: makeSelector(page, getId),
    within: (name: string) => within(page, name),
    getRole: (
      role: Roles,
      name?: string | RegExp,
      options?: GetByRoleOptions & { within?: Locator },
    ) => getRole(options?.within || page, role, name, options),
  };
}

function makeSelector(
  page: Page,
  fn: (context: Page | Locator, name: string | RegExp) => Locator,
) {
  return <T extends string | RegExp>(name: T, locator?: Locator) =>
    fn(locator || page, name);
}

function getText(
  context: Page | Locator,
  name: string | RegExp,
  options?: GetByTextOptions,
) {
  return context.getByText(name, options);
}

function getButton(context: Page | Locator, name: string | RegExp) {
  return getRole(context, "button", name);
}

function getLink(context: Page | Locator, name: string | RegExp) {
  return context.getByRole("link", { name });
}

function getField(context: Page | Locator, name: string | RegExp) {
  return context.getByLabel(name, { exact: true });
}

function getId(context: Page | Locator, testid: string | RegExp) {
  return context.getByTestId(testid);
}

function within(context: Page, testid: string) {
  return context.locator(`data-testid=${testid}`);
}

function getRole(
  context: Page | Locator,
  role: Roles,
  name?: string | RegExp,
  options?: GetByRoleOptions,
) {
  return context.getByRole(role, { ...options, ...(name ? { name } : null) });
}

type GetByTextOptions = Parameters<Page["getByText"]>["1"];
type GetByRoleOptions = Parameters<Page["getByRole"]>["1"];
type Roles = Parameters<Page["getByRole"]>["0"];
