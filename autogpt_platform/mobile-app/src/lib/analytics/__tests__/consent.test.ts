import * as SecureStore from "expo-secure-store";
import Storage from "expo-sqlite/kv-store";

import { readConsent, saveConsent } from "../consent";

jest.mock("expo-secure-store", () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
}));
jest.mock("expo-sqlite/kv-store", () => ({
  __esModule: true,
  default: { getItemSync: jest.fn(), setItemSync: jest.fn() },
}));

beforeEach(() => {
  jest.resetAllMocks();
  jest.mocked(SecureStore.getItem).mockReturnValue("allowed");
  jest.mocked(Storage.getItemSync).mockReturnValue("allowed");
});

test("a decline survives a failed SecureStore write and an app restart", () => {
  jest.mocked(SecureStore.setItem).mockImplementation(() => {
    throw new Error("SecureStore unavailable");
  });
  jest.mocked(Storage.setItemSync).mockImplementation((_key, value) => {
    if (typeof value !== "string") throw new Error("Expected a stored choice");
    jest.mocked(Storage.getItemSync).mockReturnValue(value);
  });

  expect(saveConsent("declined")).toBe(true);
  expect(SecureStore.getItem("autogpt.analyticsConsent")).toBe("allowed");
  expect(readConsent()).toBe("declined");
  jest.mocked(SecureStore.getItem).mockImplementationOnce(() => {
    throw new Error("SecureStore unavailable");
  });
  expect(readConsent()).toBe("declined");
});

test("a decline survives a failed SQLite write", () => {
  jest.mocked(Storage.setItemSync).mockImplementation(() => {
    throw new Error("SQLite unavailable");
  });
  jest.mocked(SecureStore.setItem).mockImplementation((_key, value) => {
    jest.mocked(SecureStore.getItem).mockReturnValue(value);
  });

  expect(saveConsent("declined")).toBe(true);
  expect(readConsent()).toBe("declined");
});

test("startup requires both stores to record opt-in", () => {
  jest.mocked(Storage.getItemSync).mockReturnValue(null);
  expect(readConsent()).toBe("unset");
});
