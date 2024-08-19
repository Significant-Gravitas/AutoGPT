import { FC } from "react";
import { Button } from "@/components/ui/button";
import { useSupabase } from "@/components/SupabaseProvider";

export const LinkGoogleDriveButton: FC = () =>  {
  const { supabase, isLoading } = useSupabase();

  const linkGoogleDrive = async () => {
    if (isLoading || !supabase) {
      return;
    }

    const popup = window.open(
      '/api/auth/google',
      'googleOAuth',
      'width=500,height=600'
    );

    if (!popup || popup.closed || typeof popup.closed === 'undefined') {
      console.error('Popup blocked or not created');
      return;
    }

    // Polling to check when the popup is closed
    const pollTimer = window.setInterval(async () => {
      if (popup.closed) {
        window.clearInterval(pollTimer);
        // Optionally, you can refresh tokens or check the state after OAuth is done
        const { data, error } = await supabase.auth.getSession();
        if (data.session) {
          // Tokens should now be stored in the Supabase database
          console.log('Google Drive linked successfully!');
        } else if (error) {
          console.error('Error fetching session:', error);
        }
      }
    }, 500);
  };

  return <Button onClick={linkGoogleDrive} disabled={isLoading}>Link Google Drive</Button>;
}
