import {
  createContext,
  startTransition,
  useContext,
  useEffect,
  useLayoutEffect,
  useState,
} from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { WebSocketNotification } from "@/lib/autogpt-server-api";
import OnboardingProvider, { useOnboarding } from "../onboarding-provider";

// Next commits a navigation in a transition and writes the URL during that
// commit, but React runs the transition's passive effects in a later task.
// An answer that lands in between must be decided for the new page. These run
// on a real React root, not act(), which would flush those effects at once.

const { routerReplace, router, api } = vi.hoisted(() => {
  const replace = vi.fn();
  return {
    routerReplace: replace,
    router: { replace, push: () => {}, refresh: () => {} },
    api: {
      onWebSocketMessage:
        (_type: string, _handler: (n: WebSocketNotification) => void) =>
        () => {},
      connectWebSocket: () => {},
    },
  };
});

// The app's pathname reaches the provider in the same render as the route
// change, as Next's own usePathname does.
const PathnameContext = createContext("/marketplace");
vi.mock("next/navigation", () => ({
  useRouter: () => router,
  usePathname: () => useContext(PathnameContext),
  useSearchParams: () => new URLSearchParams(window.location.search),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: true,
    isUserLoading: false,
    user: { id: "user-a" },
  }),
}));

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => api,
}));

vi.mock("@/hooks/useOnboardingTimezoneDetection", () => ({
  useOnboardingTimezoneDetection: () => undefined,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: () => {} }),
}));

const backend = vi.hoisted(() => ({
  isCompleted: false,
  completionChecks: 0,
  heldCheck: null as Promise<void> | null,
}));

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1CheckIfOnboardingIsCompleted: async () => {
    backend.completionChecks += 1;
    const isCompleted = backend.isCompleted;
    const hold = backend.heldCheck;
    backend.heldCheck = null;
    if (hold) await hold;
    return { status: 200, data: { is_completed: isCompleted } };
  },
  getV1OnboardingState: async () => ({
    status: 200,
    data: {
      completedSteps: backend.isCompleted ? ["ONBOARDING_COMPLETE"] : [],
    },
  }),
  patchV1UpdateOnboardingState: () => Promise.resolve({ status: 200 }),
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
}));

let navigate: (path: string) => void = () => {};
let releaseCheck = () => {};
let releaseOnCommitOf = "";
let releasedOn: string | null = null;
let completion: boolean | null = null;
const settledPaths: string[] = [];

// Writes the URL during the commit, as Next's history updater does, and lets
// the held answer go in the commit of the page under test.
function CommitProbe({ path }: { path: string }) {
  const { isOnboardingComplete } = useOnboarding();
  useLayoutEffect(() => {
    completion = isOnboardingComplete;
  }, [isOnboardingComplete]);
  useLayoutEffect(() => {
    window.history.replaceState(null, "", path);
    if (path.split("?")[0] === releaseOnCommitOf) {
      releasedOn = path;
      releaseCheck();
    }
  }, [path]);
  useEffect(() => {
    settledPaths.push(path);
  }, [path]);
  return null;
}

function App({ initialPath }: { initialPath: string }) {
  const [path, setPath] = useState(initialPath);
  useEffect(() => {
    navigate = (next) => startTransition(() => setPath(next));
  }, []);
  return (
    <PathnameContext.Provider value={path.split("?")[0]}>
      <OnboardingProvider>
        <CommitProbe path={path} />
      </OnboardingProvider>
    </PathnameContext.Provider>
  );
}

async function until(label: string, check: () => boolean) {
  for (let waited = 0; !check(); waited += 10) {
    if (waited > 5_000) throw new Error(`timed out waiting for ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function holdCheckUntilCommitOf(path: string) {
  releaseOnCommitOf = path;
  backend.heldCheck = new Promise<void>((resolve) => {
    releaseCheck = resolve;
  });
}

const actEnvironment = globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean };

describe("OnboardingProvider: an answer that lands during a navigation's commit", () => {
  let root: Root;
  let container: HTMLDivElement;
  let wasActEnvironment: boolean | undefined;

  beforeEach(() => {
    wasActEnvironment = actEnvironment.IS_REACT_ACT_ENVIRONMENT;
    actEnvironment.IS_REACT_ACT_ENVIRONMENT = false;
    routerReplace.mockClear();
    backend.isCompleted = false;
    backend.completionChecks = 0;
    backend.heldCheck = null;
    releasedOn = null;
    completion = null;
    settledPaths.length = 0;
    window.history.replaceState(null, "", "/marketplace");
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    releaseCheck();
    root.unmount();
    container.remove();
    window.history.replaceState(null, "", "/");
    actEnvironment.IS_REACT_ACT_ENVIRONMENT = wasActEnvironment;
  });

  test("an unfinished user who opens an account link keeps it", async () => {
    holdCheckUntilCommitOf("/link/abc");
    root.render(<App initialPath="/marketplace" />);
    await until("the completion check", () => backend.completionChecks === 1);

    navigate("/link/abc");
    await until("the answer", () => completion === false);

    expect(releasedOn).toBe("/link/abc");
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("a finished user that the login page sends on to its next is not sent Home", async () => {
    backend.isCompleted = true;
    holdCheckUntilCommitOf("/library");
    root.render(<App initialPath="/marketplace" />);
    await until("the completion check", () => backend.completionChecks === 1);

    navigate("/login?next=%2Flibrary");
    await until("the sign-in page", () =>
      settledPaths.includes("/login?next=%2Flibrary"),
    );
    // The login page follows its own deep link.
    navigate("/library");
    await until("the answer", () => completion === true);

    expect(releasedOn).toBe("/library");
    expect(routerReplace).not.toHaveBeenCalledWith("/home");
  });
});
