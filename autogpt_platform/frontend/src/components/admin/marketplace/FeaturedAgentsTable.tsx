// "use client";

// import { Button } from "@/components/ui/button";
// import { Checkbox } from "@/components/ui/checkbox";
// import { DataTable } from "@/components/ui/data-table";
// import { Agent } from "@/lib/marketplace-api";
// import { ColumnDef } from "@tanstack/react-table";
// import { ArrowUpDown } from "lucide-react";
// import { removeFeaturedAgent } from "./actions";
// import { GlobalActions } from "@/components/ui/data-table";

// export const columns: ColumnDef<Agent>[] = [
//   {
//     id: "select",
//     header: ({ table }) => (
//       <Checkbox
//         checked={
//           table.getIsAllPageRowsSelected() ||
//           (table.getIsSomePageRowsSelected() && "indeterminate")
//         }
//         onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
//         aria-label="Select all"
//       />
//     ),
//     cell: ({ row }) => (
//       <Checkbox
//         checked={row.getIsSelected()}
//         onCheckedChange={(value) => row.toggleSelected(!!value)}
//         aria-label="Select row"
//       />
//     ),
//   },
//   {
//     header: ({ column }) => {
//       return (
//         <Button
//           variant="ghost"
//           onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
//         >
//           Name
//           <ArrowUpDown className="ml-2 h-4 w-4" />
//         </Button>
//       );
//     },
//     accessorKey: "name",
//   },
//   {
//     header: "Description",
//     accessorKey: "description",
//   },
//   {
//     header: "Categories",
//     accessorKey: "categories",
//   },
//   {
//     header: "Keywords",
//     accessorKey: "keywords",
//   },
//   {
//     header: "Downloads",
//     accessorKey: "downloads",
//   },
//   {
//     header: "Author",
//     accessorKey: "author",
//   },
//   {
//     header: "Version",
//     accessorKey: "version",
//   },
//   {
//     header: "actions",
//     cell: ({ row }) => {
//       const handleRemove = async () => {
//         await removeFeaturedAgentWithId();
//       };
//       // const handleEdit = async () => {
//       //   console.log("edit");
//       // };
//       const removeFeaturedAgentWithId = removeFeaturedAgent.bind(
//         null,
//         row.original.id,
//       );
//       return (
//         <div className="flex justify-end gap-2">
//           <Button variant="outline" size="sm" onClick={handleRemove}>
//             Remove
//           </Button>
//           {/* <Button variant="outline" size="sm" onClick={handleEdit}>
//             Edit
//           </Button> */}
//         </div>
//       );
//     },
//   },
// ];

// export default function FeaturedAgentsTable({
//   agents,
//   globalActions,
// }: {
//   agents: Agent[];
//   globalActions: GlobalActions<Agent>[];
// }) {
//   return (
//     <DataTable
//       columns={columns}
//       data={agents}
//       filterPlaceholder="Search agents..."
//       filterColumn="name"
//       globalActions={globalActions}
//     />
//   );
// }
