import { LibraryAgentCard2 } from "../LibraryAgentCard2/LibraryAgentCard2";

export const LibraryAgentList2 = () => {
  const agents = [
    {
      id: "1",
      title: "Web Scraping Assistant",
      imageUrl: "/placeholder.png",
      lastRunTime: "2 hours ago",
      totalRuns: 156,
      runningAgents: 2,
      source: "From Marketplace",
    },
    {
      id: "2",
      title: "Data Analysis Helper",
      imageUrl: "/placeholder.png",
      lastRunTime: "5 hours ago",
      totalRuns: 89,
      runningAgents: 1,
      source: "Built by you",
    },
    {
      id: "3",
      title: "Content Generator",
      imageUrl: "/placeholder.png",
      lastRunTime: "1 day ago",
      totalRuns: 234,
      runningAgents: 3,
      source: "From Marketplace",
    },
    {
      id: "4",
      title: "Code Review Bot",
      imageUrl: "/placeholder.png",
      lastRunTime: "3 days ago",
      totalRuns: 67,
      runningAgents: 0,
      source: "Built by you",
    },
    {
      id: "5",
      title: "Social Media Manager",
      imageUrl: "/placeholder.png",
      lastRunTime: "1 week ago",
      totalRuns: 445,
      runningAgents: 1,
      source: "From Marketplace",
    },
    {
      id: "6",
      title: "Customer Support Agent",
      imageUrl: "/placeholder.png",
      lastRunTime: "2 weeks ago",
      totalRuns: 178,
      runningAgents: 0,
      source: "Built by you",
    },
  ];
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
      {agents.map((agent) => (
        <LibraryAgentCard2
          key={agent.id}
          id={agent.id}
          title={agent.title}
          imageUrl={agent.imageUrl}
          lastRunTime={agent.lastRunTime}
          totalRuns={agent.totalRuns}
          runningAgents={agent.runningAgents}
          source={agent.source}
        />
      ))}
    </div>
  );
};
