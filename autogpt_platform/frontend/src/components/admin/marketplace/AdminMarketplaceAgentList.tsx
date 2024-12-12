// import { Agent } from "@/lib/marketplace-api";
// import AdminMarketplaceCard from "./AdminMarketplaceCard";
// import { ClipboardX } from "lucide-react";

// export default function AdminMarketplaceAgentList({
//   agents,
//   className,
// }: {
//   agents: Agent[];
//   className?: string;
// }) {
//   if (agents.length === 0) {
//     return (
//       <div className={className}>
//         <h3 className="text-lg font-semibold">Agents to review</h3>
//         <div className="flex flex-col items-center justify-center py-12 text-gray-500">
//           <ClipboardX size={48} />
//           <p className="mt-4 text-lg font-semibold">No agents to review</p>
//         </div>
//       </div>
//     );
//   }

//   return (
//     <div className={`flex flex-col gap-4 ${className}`}>
//       <div>
//         <h3 className="text-lg font-semibold">Agents to review</h3>
//       </div>
//       <div className="flex flex-col gap-4">
//         {agents.map((agent) => (
//           <AdminMarketplaceCard agent={agent} key={agent.id} />
//         ))}
//       </div>
//     </div>
//   );
// }
