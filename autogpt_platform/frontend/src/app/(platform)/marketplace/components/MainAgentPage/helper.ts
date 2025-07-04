export const getBreadcrumbs = (agent: {
  creator: string;
  agent_name: string;
}) => {
  const breadcrumbs = [
    { name: "Marketplace", link: "/marketplace" },
    {
      name: agent.creator,
      link: `/marketplace/creator/${encodeURIComponent(agent.creator)}`,
    },
    { name: agent.agent_name, link: "#" },
  ];

  return breadcrumbs;
};
