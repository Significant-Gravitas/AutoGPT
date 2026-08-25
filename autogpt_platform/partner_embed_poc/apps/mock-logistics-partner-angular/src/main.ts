import { provideZonelessChangeDetection } from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import "@autogpt/embedded-chat-element";

import { App } from "./app/app";

bootstrapApplication(App, {
  providers: [provideZonelessChangeDetection()],
}).catch((error: unknown) => {
  console.error(error);
});
