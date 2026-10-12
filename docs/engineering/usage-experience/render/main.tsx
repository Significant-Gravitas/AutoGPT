import { useState, type ComponentProps } from "react";
import { createRoot } from "react-dom/client";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import SettingsBillingPage from "@/app/(platform)/settings/billing/page";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatComposerFrame } from "@/app/(platform)/copilot/components/ChatContainer/components/ChatComposerFrame";
import { UsageLimitReachedCard } from "@/app/(platform)/copilot/components/UsageLimits/UsageLimitReachedCard/UsageLimitReachedCard";
import { RateLimitResetDialog } from "@/app/(platform)/copilot/components/RateLimitResetDialog/RateLimitResetDialog";
import { SwitchTierDialog } from "@/app/(platform)/settings/billing/components/SubscriptionTab/YourPlanCard/SwitchTierDialog";
import { PaywallModal } from "@/app/(platform)/PaywallGate/PaywallModal";
import { ProActivationDialog } from "@/components/organisms/ProActivation/ProActivationDialog";
import { useUsageActions } from "@/services/usageExperience/useUsageActions";
import {
  fixture,
  setFixture,
  noop,
  type Experience,
  activationTerms,
  upgradeReview,
} from "./fixtures";

type Screen =
  | "dialog"
  | "chat"
  | "notice"
  | "billing"
  | "paywall"
  | "review"
  | "activation"
  | "pending"
  | "ready";
declare global {
  interface Window {
    __implementationPreview: {
      show: (experience: Experience, screen: Screen) => void;
      get: () => { experience: Experience; screen: Screen };
    };
  }
}

function Composer() {
  const paused = !["active", "fresh"].includes(fixture.state);
  return (
    <div data-capture="chat" className="mx-auto w-full max-w-3xl">
      <ChatComposerFrame
        paused={paused}
        notice={
          paused ? <UsageLimitReachedCard sessionID="render-session" /> : null
        }
      >
        <ChatInput
          sendDisabled={paused}
          hasSession
          sessionId="render-session"
          draftKey="session:render-session"
          inputId="chat-input-session"
          onSend={noop}
        />
      </ChatComposerFrame>
    </div>
  );
}

const queryClient = new QueryClient();
function App() {
  const [screen, setScreen] = useState<Screen>("billing");
  const [version, setVersion] = useState(0);
  const [open, setOpen] = useState(true);
  const actions = useUsageActions();
  window.__implementationPreview = {
    show(experience, nextScreen) {
      setFixture(experience);
      sessionStorage.setItem(
        "autogpt:chat-draft:fixture-user:session%3Arender-session",
        "Turn these findings into a launch plan.",
      );
      setScreen(nextScreen);
      setOpen(true);
      setVersion((value) => value + 1);
    },
    get() {
      return { experience: fixture, screen };
    },
  };
  const activation: ComponentProps<typeof ProActivationDialog>["activation"] = {
    attempt: {
      activation_id: "fixture-activation",
      id: "fixture-activation",
      usage_reset: screen === "ready",
      status:
        screen === "activation"
          ? "confirmation_required"
          : screen === "ready"
            ? "ready"
            : "processing",
      return_to: "/copilot",
      terms: screen === "activation" ? activationTerms : undefined,
      terms_token: screen === "activation" ? "f".repeat(64) : undefined,
    },
    isOpen: open,
    setOpen,
    isBusy: false,
    isReady: screen === "ready",
    readyPlan: screen === "ready" ? "PRO" : null,
    hasFreshUsage: screen === "ready",
    error: "",
    needsNewTerms: false,
    retryConsent: false,
    confirm: async () => {},
    check: async () => {},
    start: async () => {},
  };
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <div
          key={version}
          className="flex min-h-screen items-center justify-center bg-white"
        >
          {screen === "billing" && (
            <div className="w-full py-7" data-capture="billing">
              <SettingsBillingPage />
            </div>
          )}
          {["chat", "pending", "ready"].includes(screen) && <Composer />}
          {screen === "notice" && (
            <div className="w-full max-w-3xl p-3" data-capture="notice">
              <UsageLimitReachedCard sessionID="render-session" />
            </div>
          )}
          {screen === "dialog" && (
            <RateLimitResetDialog
              isOpen={open}
              onClose={() => setOpen(false)}
              experience={actions.experience}
              offer={actions.offer}
              onUpgrade={noop}
              unavailable={fixture.state === "unavailable"}
              onRetry={noop}
              alternative={
                fixture.linkedProvider ? { display_name: "Claude" } : undefined
              }
              onContinue={fixture.linkedProvider ? noop : undefined}
            />
          )}
          {screen === "review" && (
            <SwitchTierDialog
              isOpen={open}
              onOpenChange={setOpen}
              targetTierLabel="Max"
              priceCents={32000}
              billingCycle="monthly"
              usageCarriesOver
              usageMultiplier={8.5}
              upgradeReview={upgradeReview}
              body=""
              isSaving={false}
              onConfirm={noop}
            />
          )}
          {screen === "paywall" && <PaywallModal />}
          {["activation", "pending", "ready"].includes(screen) && (
            <ProActivationDialog activation={activation} />
          )}
        </div>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

createRoot(document.getElementById("autogpt-system-redesign")!).render(<App />);
