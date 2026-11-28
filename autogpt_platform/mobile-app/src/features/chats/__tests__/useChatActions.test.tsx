import { act, renderHook } from "@testing-library/react-native";
import { router } from "expo-router";
import { Alert } from "react-native";

import { chooseAction } from "@/lib/actionMenu";
import { chatKeys, deleteSession, setSessionPinned } from "@/lib/api/chat";
import { queryClient, sessionGeneration } from "@/lib/session";

import { useChatActions } from "../useChatActions";

const mockCurrentRoute = jest.fn();
jest.mock("expo-router", () => ({
  router: { navigate: jest.fn() },
  useNavigationContainerRef: () => ({ getCurrentRoute: mockCurrentRoute }),
}));
jest.mock("@/lib/actionMenu", () => ({ chooseAction: jest.fn() }));
jest.mock("@/lib/haptics", () => ({ tapMenu: jest.fn() }));
jest.mock("@/lib/api/chat", () => ({
  chatKeys: { sessions: ["sessions"], session: (id: string) => ["session", id] },
  deleteSession: jest.fn(),
  setSessionPinned: jest.fn(),
}));
jest.mock("@/lib/session", () => {
  const { QueryClient } = jest.requireActual("@tanstack/react-query");
  return { queryClient: new QueryClient(), sessionGeneration: jest.fn() };
});

const chat = {
  id: "chat-a",
  title: "First chat",
  is_pinned: false,
  updated_at: "2026-11-01T00:00:00Z",
  created_at: "2026-11-01T00:00:00Z",
  is_processing: false,
  expert_id: null,
};

beforeEach(() => {
  jest.mocked(sessionGeneration).mockReturnValue(1);
  queryClient.setQueryData(chatKeys.sessions, { sessions: [chat], total: 1 });
});

afterEach(() => {
  queryClient.clear();
  jest.clearAllMocks();
  jest.restoreAllMocks();
});

test("a late rollback cannot put the prior account's chats back in the cache", async () => {
  let fail = () => {};
  jest.mocked(chooseAction).mockResolvedValue("pin");
  jest.mocked(setSessionPinned).mockImplementation(
    () => new Promise((_resolve, reject) => {
      fail = () => reject(new Error("request failed"));
    }),
  );
  const { result } = await renderHook(useChatActions);
  await act(() => result.current.openMenu(chat));
  expect(setSessionPinned).toHaveBeenCalledWith(chat.id, true);

  const nextAccount = { sessions: [{ ...chat, id: "chat-b", title: "Other account" }], total: 1 };
  jest.mocked(sessionGeneration).mockReturnValue(2);
  queryClient.clear();
  queryClient.setQueryData(chatKeys.sessions, nextAccount);
  await act(async () => fail());

  expect(queryClient.getQueryData(chatKeys.sessions)).toEqual(nextAccount);
});

test("a failed pin leaves a successful pin on another chat in place", async () => {
  let fail = () => {};
  jest.mocked(chooseAction).mockResolvedValue("pin");
  jest.mocked(setSessionPinned)
    .mockImplementationOnce(
      () => new Promise((_resolve, reject) => {
        fail = () => reject(new Error("request failed"));
      }),
    )
    .mockResolvedValueOnce({ status: "ok" });
  const otherChat = { ...chat, id: "chat-b", title: "Second chat" };
  queryClient.setQueryData(chatKeys.sessions, { sessions: [chat, otherChat], total: 2 });
  const { result } = await renderHook(useChatActions);
  await act(() => result.current.openMenu(chat));
  await act(() => result.current.openMenu(otherChat));
  expect(setSessionPinned).toHaveBeenNthCalledWith(2, otherChat.id, true);
  await act(async () => fail());

  expect(queryClient.getQueryData(chatKeys.sessions)).toEqual({
    sessions: [{ ...otherChat, is_pinned: true }, chat],
    total: 2,
  });
});

test.each(["chat-a", "chat-b"])("checks the active chat %s after deleting", async (activeID) => {
  let finish = () => {};
  jest.mocked(chooseAction).mockResolvedValue("delete");
  jest.spyOn(Alert, "alert").mockImplementation((_title, _message, buttons) => {
    buttons?.find((button) => button.style === "destructive")?.onPress?.();
  });
  jest.mocked(deleteSession).mockImplementation(
    () => new Promise<void>((resolve) => { finish = resolve; }),
  );
  mockCurrentRoute.mockReturnValue({ name: "chat/[sessionId]", params: { sessionId: "chat-a" } });
  const { result } = await renderHook(useChatActions);
  await act(() => result.current.openMenu(chat));
  expect(deleteSession).toHaveBeenCalledWith(chat.id);

  mockCurrentRoute.mockReturnValue({ name: "chat/[sessionId]", params: { sessionId: activeID } });
  await act(async () => finish());
  expect(router.navigate).toHaveBeenCalledTimes(activeID === chat.id ? 1 : 0);
});
