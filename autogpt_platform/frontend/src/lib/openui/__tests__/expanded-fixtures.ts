export const places = `root = Workspace("Chicago field visits", "Illustrative visit plan using supplied coordinates", [places])
places = Map("Visit locations", "Select a place to explore it in this chat.", [{name: "River North", latitude: 41.8924, longitude: -87.6341, detail: "Morning customer visit", category: "Customer"}, {name: "West Loop", latitude: 41.8825, longitude: -87.6441, detail: "Afternoon partner visit", category: "Partner"}])`;

export const planning = `root = Workspace("Visit planning", "Fictional planning data", [schedule, trend, mix, brief])
schedule = Timeline("Visit schedule", [{time: "09:00", title: "Preparation", detail: "Review the brief", status: "done"}, {time: "10:00", title: "Customer visit", detail: "Discuss the next milestone", status: "current"}, {time: "14:00", title: "Partner visit", detail: "Compare the options", status: "planned"}])
trend = TrendChart("Weekly change", "Fictional observations", "leads", [{label: "Week 1", value: -2}, {label: "Week 2", value: 4}, {label: "Week 3", value: 8}])
mix = DonutChart("Visit mix", "Fictional visit breakdown", "visits", [{label: "Customers", value: 3}, {label: "Partners", value: 1}])
brief = Form("visit", "Adjust this plan", [mode, date, budget], "Update plan", "Revise the visit plan using these preferences")
mode = SelectField("mode", "Travel mode", "walking", [{label: "Walking", value: "walking"}, {label: "Transit", value: "transit"}])
date = DateField("date", "Visit date", "2026-10-09")
budget = NumberField("budget", "Budget", 100, 0, 1000, 10)`;
