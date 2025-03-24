

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { PaginationControls } from "../../ui/pagination-controls";
import { getGrantHistory, getUserBalances } from "@/app/admin/spending/actions";
import { AddCreditButton } from "./add-credit-button";


export async function AdminUserSpendingBalances({
  initialPage = 1,
  initialSearch,
}: {
  initialPage?: number;
  initialSearch?: string;
}) {
  // Server-side data fetching
  const { balances, pagination } = await getUserBalances(
    initialPage,
    10,
    initialSearch,
  );

  return (
    <>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Email</TableHead>
              <TableHead>Current Balance</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {balances.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="py-10 text-center">
                  No users found
                </TableCell>
              </TableRow>
            ) : (
              balances.map((balance) => (
                <TableRow key={balance.user_id}>
                  <TableCell className="font-medium">
                    {balance.user_email}
                  </TableCell>
                  <TableCell>{balance.balance}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <AddCreditButton userId={balance.user_id} userEmail={balance.user_email} currentBalance={balance.balance} />
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )
            }
          </TableBody>
        </Table>
      </div>

      <PaginationControls
        currentPage={initialPage}
        totalPages={pagination.total_pages}
        pathParam="userPage"
      />
    </ >
  );
}
