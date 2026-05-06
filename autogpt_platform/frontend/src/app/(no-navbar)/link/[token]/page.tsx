"use client";

import { ErrorView } from "./components/ErrorView";
import { LoadingView } from "./components/LoadingView";
import { NotAuthenticatedView } from "./components/NotAuthenticatedView";
import { ReadyView } from "./components/ReadyView";
import { SuccessView } from "./components/SuccessView";
import { usePlatformLinkingPage } from "./usePlatformLinkingPage";

export default function PlatformLinkPage() {
  const page = usePlatformLinkingPage();

  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      {page.status === "loading" && <LoadingView />}

      {page.status === "not-authenticated" && page.token && (
        <NotAuthenticatedView
          token={page.token}
          loginRedirect={page.loginRedirect}
        />
      )}

      {page.status === "ready" && page.viewData && (
        <ReadyView
          linkType={page.viewData.linkType}
          platform={page.viewData.platform}
          serverName={page.viewData.serverName}
          userEmail={page.userEmail}
          isLinking={false}
          onLink={page.handleLink}
          onSwitchAccount={page.handleSwitchAccount}
        />
      )}

      {page.status === "linking" && page.viewData && (
        <ReadyView
          linkType={page.viewData.linkType}
          platform={page.viewData.platform}
          serverName={page.viewData.serverName}
          userEmail={page.userEmail}
          isLinking
          onLink={page.handleLink}
          onSwitchAccount={page.handleSwitchAccount}
        />
      )}

      {page.status === "success" && page.successData && (
        <SuccessView
          linkType={page.successData.linkType}
          platform={page.successData.platform}
          serverName={page.successData.serverName}
        />
      )}

      {page.status === "error" && <ErrorView message={page.errorMessage} />}

      <div className="mt-8 text-center text-xs text-muted-foreground">
        <p>Powered by AutoGPT Platform</p>
      </div>
    </div>
  );
}
