import React, { useState, useEffect } from "react";
import Integration from "../Integration";
import { integrationsListData } from "../../testing_data";

interface IntegrationListProps {
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
}

export interface IntegrationData {
  title: string;
  icon_url: string;
  description: string;
  number_of_blocks: number;
}

const IntegrationList: React.FC<IntegrationListProps> = ({
  setIntegration,
}) => {
  const [integrations, setIntegrations] = useState<IntegrationData[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Update Block Menu fetching
  useEffect(() => {
    const fetchIntegrations = async () => {
      setIsLoading(true);
      try {
        await new Promise((resolve) => setTimeout(resolve, 1000));

        setIntegrations(integrationsListData);
      } catch (error) {
        console.error("Failed to fetch integrations:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchIntegrations();
  }, []);

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
          title={integration.title}
          icon_url={integration.icon_url}
          description={integration.description}
          number_of_blocks={integration.number_of_blocks}
          onClick={() => setIntegration(integration.title)}
        />
      ))}
    </div>
  );
};

export default IntegrationList;
