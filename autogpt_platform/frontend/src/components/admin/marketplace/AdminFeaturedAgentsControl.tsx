// import { Button } from "@/components/ui/button";
// import {
//   getFeaturedAgents,
//   removeFeaturedAgent,
//   getCategories,
//   getNotFeaturedAgents,
// } from "./actions";

// import FeaturedAgentsTable from "./FeaturedAgentsTable";
// import { AdminAddFeaturedAgentDialog } from "./AdminAddFeaturedAgentDialog";
// import { revalidatePath } from "next/cache";
// import * as Sentry from "@sentry/nextjs";

// export default async function AdminFeaturedAgentsControl({
//   className,
// }: {
//   className?: string;
// }) {
//   // add featured agent button
//   //   modal to select agent?
//   //   modal to select categories?
//   // table of featured agents
//   // in table
//   //    remove featured agent button
//   //    edit featured agent categories button
//   // table footer
//   //    Next page button
//   //    Previous page button
//   //    Page number input
//   //    Page size input
//   //    Total pages input
//   //    Go to page button

//   const page = 1;
//   const pageSize = 10;

//   const agents = await getFeaturedAgents(page, pageSize);

//   const categories = await getCategories();

//   const notFeaturedAgents = await getNotFeaturedAgents();

//   return (
//     <div className={`flex flex-col gap-4 ${className}`}>
//       <div className="mb-4 flex justify-between">
//         <h3 className="text-lg font-semibold">Featured Agent Controls</h3>
//         <AdminAddFeaturedAgentDialog
//           categories={categories.unique_categories}
//           agents={notFeaturedAgents.items}
//         />
//       </div>
//       <FeaturedAgentsTable
//         agents={agents.items}
//         globalActions={[
//           {
//             component: <Button>Remove</Button>,
//             action: async (rows) => {
//               "use server";
//               return await Sentry.withServerActionInstrumentation(
//                 "removeFeaturedAgent",
//                 {},
//                 async () => {
//                   const all = rows.map((row) => removeFeaturedAgent(row.id));
//                   await Promise.all(all);
//                   revalidatePath("/marketplace");
//                 },
//               );
//             },
//           },
//         ]}
//       />
//     </div>
//   );
// }
