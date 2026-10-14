export const connected = `root = Workspace("Choose a weekend", "Illustrative estimates, not bookings", [brief])
brief = Form("trip", "Make it yours", [choice, guests, notes, access, interests], "Continue with this plan", "Update the trip using my choices.")
choice = Comparison("stay", "Where would you stay?", "", [{id:"city",title:"City hotel",description:"Near the museums; amount is per night",facts:[{label:"Tradeoff",value:"More noise"}],amount:180},{id:"coast",title:"Coastal cabin",description:"A quiet base; amount is per night",facts:[{label:"Tradeoff",value:"Longer journey"}],amount:140}])
guests = NumberField("nights", "Nights", 1, 1, 6, 1)
notes = TextAreaField("notes", "Anything else?", "", "Optional details", false)
access = ToggleField("step_free", "Step-free access needed", false)
interests = MultiSelectField("interests", "Interests", ["art"], [{label:"Art",value:"art"},{label:"History",value:"history"}], false)`;
