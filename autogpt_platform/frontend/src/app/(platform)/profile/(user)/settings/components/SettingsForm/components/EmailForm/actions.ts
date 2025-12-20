import { getServerAuthToken } from "@/lib/auth/server/getServerAuth";
import { environment } from "@/services/environment";

export async function updateUserEmail(email: string) {
  const token = await getServerAuthToken();

  if (!token) {
    return { data: null, error: { message: "Not authenticated" } };
  }

  try {
    const response = await fetch(
      `${environment.getAGPTServerBaseUrl()}/api/auth/update-email`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ email }),
      },
    );

    if (!response.ok) {
      const data = await response.json();
      return {
        data: null,
        error: { message: data.detail || "Failed to update email" },
      };
    }

    const data = await response.json();
    return { data, error: null };
  } catch (error) {
    console.error("Error updating email:", error);
    return { data: null, error: { message: "Failed to update email" } };
  }
}
