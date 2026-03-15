"use client";

import React, { useState } from "react";

import { Key } from "lucide-react";
import { getV1GetAyrshareSsoUrl } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Button } from "@/components/atoms/Button/Button";

// This SSO button is not a part of inputSchema - that's why we are not rendering it using Input renderer
export const AyrshareConnectButton = () => {
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleSSOLogin = async () => {
    setIsLoading(true);
    try {
      const { data, status } = await getV1GetAyrshareSsoUrl();
      if (status !== 200) {
        throw new Error(data.detail);
      }
      const popup = window.open(data.sso_url, "_blank", "popup=true");
      if (!popup) {
        throw new Error(
          "Please allow popups for this site to be able to login with Ayrshare",
        );
      }
      toast({
        title: "Success",
        description: "Please complete the authentication in the popup window",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: `Error getting SSO URL: ${error}`,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    // TODO :Need better UI to show user which social media accounts are connected
    <div className="mt-4 flex flex-col gap-2 px-4">
      <Button
        type="button"
        onClick={handleSSOLogin}
        disabled={isLoading}
        className="h-fit w-full py-2"
        loading={isLoading}
        leftIcon={<Key className="mr-2 h-4 w-4" />}
      >
        Connect Social Media Accounts
      </Button>
    </div>
  );
};
