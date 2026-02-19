import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, RenderOptions } from "@testing-library/react";
import { ReactElement, ReactNode } from "react";
import {
  MockOnboardingProvider,
  useOnboarding as mockUseOnboarding,
} from "./helpers/mock-onboarding-provider";

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: mockUseOnboarding,
  default: vi.fn(),
}));

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        refetchOnWindowFocus: false,
        refetchOnMount: false,
        refetchOnReconnect: false,
      },
    },
  });
}

function TestProviders({ children }: { children: ReactNode }) {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <BackendAPIProvider>
        <MockOnboardingProvider>{children}</MockOnboardingProvider>
      </BackendAPIProvider>
    </QueryClientProvider>
  );
}

function customRender(
  ui: ReactElement,
  options?: Omit<RenderOptions, "wrapper">,
) {
  return render(ui, { wrapper: TestProviders, ...options });
}

export * from "@testing-library/react";
export { customRender as render };
