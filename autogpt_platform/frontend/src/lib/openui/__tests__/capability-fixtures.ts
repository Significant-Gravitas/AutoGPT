export function formView(control: string, extras = "") {
  return `root = Workspace("Review", "Supplied choices", [form])\nform = Form("review", "Your choices", [control${extras ? ", extra" : ""}], "Continue", "Use my choices")\ncontrol = ${control}${extras ? `\nextra = ${extras}` : ""}`;
}

export const tripProps = {
  name: "trip",
  title: "SF day",
  days: [
    {
      label: "Saturday",
      date: "2026-10-17",
      windows: [{ start: "09:00", end: "17:00" }],
    },
  ],
  tasks: [
    {
      id: "bridge_visit",
      title: "Golden Gate Bridge",
      placeId: "bridge",
      minutes: 60,
      done: false,
      splittable: false,
    },
    {
      id: "museum_visit",
      title: "Museum visit",
      placeId: "museum",
      minutes: 90,
      done: false,
      splittable: false,
    },
  ],
  timeZone: "America/Los_Angeles",
  locations: [
    {
      id: "bridge",
      name: "Golden Gate Bridge",
      latitude: 37.8199,
      longitude: -122.4783,
      category: "Landmark",
      detail: "Fixture location supplied for testing.",
    },
    {
      id: "museum",
      name: "Museum",
      latitude: 37.7857,
      longitude: -122.4011,
      category: "Museum",
      detail: "Fixture location supplied for testing.",
    },
  ],
  legs: [
    {
      from: "bridge",
      to: "museum",
      minutes: 45,
      mode: "Transit estimate",
      validOn: "2026-10-17",
    },
  ],
};
export const tripSource = formView(
  `Itinerary(${[tripProps.name, tripProps.title, tripProps.days, tripProps.tasks, tripProps.timeZone, tripProps.locations, tripProps.legs].map((arg) => JSON.stringify(arg)).join(", ")})`,
);
