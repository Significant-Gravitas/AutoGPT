import React, { useState, useEffect } from "react";
import Integration from "../Integration";
import { useBlockMenuContext } from "../block-menu-provider";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { Provider } from "@/lib/autogpt-server-api";

const IntegrationList: React.FC = ({}) => {
  const { setIntegration } = useBlockMenuContext();
  const [integrations, setIntegrations] = useState<Provider[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const api = useBackendAPI();

  useEffect(() => {
    const fetchIntegrations = async () => {
      setIsLoading(true);
      try {
        // Some integrations are missing, like twitter or todoist or more
        const providers = await api.getProviders();
        setIntegrations(providers.providers);
      } catch (error) {
        console.error("Failed to fetch integrations:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchIntegrations();
  }, [api]);

  if (isLoading) {
    return (
      <div className="space-y-3">
        {Array(5)
          .fill(null)
          .map((_, index) => (
            <Integration.Skeleton key={index} />
          ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {integrations.map((integration, index) => (
        <Integration
          key={index}
          title={integration.name}
          icon_url={`/integrations/${integration.name}.png`}
          description={integration.description}
          number_of_blocks={integration.integration_count}
          onClick={() => setIntegration(integration.name)}
        />
      ))}
    </div>
  );
};

export default IntegrationList;
