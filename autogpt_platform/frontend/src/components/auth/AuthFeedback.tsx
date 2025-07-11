import { HelpItem } from "@/components/auth/help-item";
import { Card, CardContent } from "@/components/ui/card";
import { BehaveAs } from "@/lib/utils";
import { AlertCircle, CheckCircle } from "lucide-react";

interface Props {
  type: "login" | "signup";
  message?: string | null;
  isError?: boolean;
  behaveAs?: BehaveAs;
}

export default function AuthFeedback({
  type,
  message = "",
  isError = false,
  behaveAs = BehaveAs.CLOUD,
}: Props) {
  const displayMessage =
    message || (isError ? "Something went wrong. Please try again." : "");

  const isCloudMode = behaveAs === BehaveAs.CLOUD;
  const isLocalMode = behaveAs === BehaveAs.LOCAL;
  const showCloudHelp = isError && isCloudMode;
  const showLocalHelp = isError && isLocalMode;
  const isSignupFlow = type === "signup";
  const hasAnyHelpContent = showCloudHelp || showLocalHelp;

  const hasContent = displayMessage || hasAnyHelpContent;

  if (!hasContent) {
    return null;
  }

  return (
    <div className="mt-4 w-full space-y-4">
      {/* Message feedback */}
      {displayMessage && (
        <div className="text-center text-sm font-medium leading-normal">
          {isError ? (
            <div className="flex items-center justify-center space-x-2 text-red-500">
              <AlertCircle className="h-4 w-4" />
              <span>{displayMessage}</span>
            </div>
          ) : (
            <div className="flex items-center justify-center space-x-2 text-green-600">
              <CheckCircle className="h-4 w-4" />
              <span>{displayMessage}</span>
            </div>
          )}
        </div>
      )}

      {/* Cloud-specific help */}
      {showCloudHelp &&
        (isSignupFlow ? (
          <Card className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
            <CardContent className="p-0">
              <div className="divide-y divide-slate-100">
                <span className="my-3 block text-center text-sm font-medium text-red-500">
                  The provided email may not be allowed to sign up.
                </span>
                <HelpItem
                  title="AutoGPT Platform is currently in closed beta. "
                  description="You can join "
                  linkText="the waitlist here"
                  href="https://agpt.co/waitlist"
                />
                <HelpItem title="Make sure you use the same email address you used to sign up for the waitlist." />
                <HelpItem
                  title="You can self host the platform!"
                  description="Visit our"
                  linkText="GitHub repository"
                  href="https://github.com/Significant-Gravitas/AutoGPT"
                />
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
            <CardContent className="p-0">
              <div className="divide-y divide-slate-100">
                <HelpItem
                  title="Having trouble logging in?"
                  description="Make sure you've already "
                  linkText="signed up"
                  href="/signup"
                />
              </div>
            </CardContent>
          </Card>
        ))}

      {/* Local-specific help */}
      {showLocalHelp && (
        <Card className="w-full overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <div className="w-full divide-y divide-slate-100">
            <HelpItem
              title="Having trouble getting AutoGPT running locally?"
              description="Ask for help on our"
              linkText="Discord"
              href="https://discord.gg/autogpt"
            />

            <HelpItem
              title="Think you've found a bug?"
              description="Open an issue on our"
              linkText="GitHub"
              href="https://github.com/Significant-Gravitas/AutoGPT"
            />

            <HelpItem
              title="Interested in the cloud-hosted version?"
              description="Join our"
              linkText="waitlist here"
              href="https://agpt.co/waitlist"
            />
          </div>
        </Card>
      )}
    </div>
  );
}
