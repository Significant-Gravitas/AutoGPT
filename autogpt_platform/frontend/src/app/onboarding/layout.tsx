"use client";
import { UserOnboarding } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { usePathname, useRouter } from "next/navigation";
import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

const OnboardingContext = createContext<
  | {
      state: UserOnboarding | null;
      setState: (state: Partial<UserOnboarding>) => void;
    }
  | undefined
>(undefined);

export function useOnboarding(step?: number) {
  const context = useContext(OnboardingContext);
  if (!context)
    throw new Error("useOnboarding must be used within /onboarding pages");

  useEffect(() => {
    if (!step) return;

    context.setState({ step });
  }, [step]);

  return context;
}

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  const [state, setStateRaw] = useState<UserOnboarding | null>(null);
  const api = useBackendAPI();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    const fetchOnboarding = async () => {
      const onboarding = await api.getUserOnboarding();
      setStateRaw(onboarding);

      // Redirect outside onboarding if completed
      if (onboarding.isCompleted && !pathname.startsWith("/onboarding/reset")) {
        router.push("/library");
      }
    };
    fetchOnboarding();
  }, [api, pathname, router]);

  const setState = useCallback(
    (newState: Partial<UserOnboarding>) => {
      function removeNullFields<T extends object>(obj: T): Partial<T> {
        return Object.fromEntries(
          Object.entries(obj).filter(([_, value]) => value != null),
        ) as Partial<T>;
      }

      const updateState = (state: Partial<UserOnboarding>) => {
        if (!state) return;

        api.updateUserOnboarding(state);
      };

      setStateRaw((prev) => {
        if (newState.step && prev && prev?.step !== newState.step) {
          updateState(removeNullFields({ ...prev, ...newState }));
        }

        if (!prev) {
          // Handle initial state
          return {
            step: 1,
            integrations: [],
            isCompleted: false,
            ...newState,
          };
        }
        return { ...prev, ...newState };
      });
    },
    [api, setStateRaw],
  );

  return (
    <OnboardingContext.Provider value={{ state, setState }}>
      <div className="flex min-h-screen w-full items-center justify-center bg-gray-100">
        <div className="mx-auto flex w-full flex-col items-center">
          {children}
        </div>
      </div>
    </OnboardingContext.Provider>
  );
}
