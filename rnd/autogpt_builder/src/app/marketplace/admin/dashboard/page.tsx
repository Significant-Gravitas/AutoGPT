// app/marketplace/admin/dashboard/page.tsx
import { withRoleAccess } from '@/lib/withRoleAccess';

async function AdminDashboard() {
  // You can fetch data here directly if needed
  // const data = await fetchAdminData();
  await withRoleAccess(['admin']);

  return (
    <div>
      <h1>Admin Dashboard</h1>
      {/* Add your admin-only content here */}
    </div>
);
}

export default AdminDashboard;