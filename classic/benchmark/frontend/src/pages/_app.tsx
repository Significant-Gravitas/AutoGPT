import { type AppType } from "next/dist/shared/lib/utils";
import "~/styles/globals.css";
import "@fortawesome/fontawesome-svg-core/styles.css";

const MyApp: AppType = ({ Component, pageProps }) => {
  return <Component {...pageProps} />;
};

export default MyApp;
