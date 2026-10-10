export const connected = `root = Workspace("Choose a weekend", "Illustrative estimates, not bookings", [brief, total])
brief = Form("trip", "Make it yours", [choice, guests, extras, notes, access, interests], "Continue with this plan", "Update the trip using my choices.")
choice = Comparison("stay", "Where would you stay?", "", [{id:"city",title:"City hotel",description:"Near the museums; amount is per night",facts:[{label:"Tradeoff",value:"More noise"}],amount:180},{id:"coast",title:"Coastal cabin",description:"A quiet base; amount is per night",facts:[{label:"Tradeoff",value:"Longer journey"}],amount:140}])
guests = NumberField("nights", "Nights", 1, 1, 6, 1)
extras = CostTable("extras", "Adjust your extras", "USD", [{id:"museum",label:"Museum tickets",quantity:2,unitPrice:25,included:true},{id:"ride",label:"Cable car rides",quantity:2,unitPrice:8,included:true}])
notes = TextAreaField("notes", "Anything else?", "", "Optional details", false)
access = ToggleField("step_free", "Step-free access needed", false)
interests = MultiSelectField("interests", "Interests", ["art"], [{label:"Art",value:"art"},{label:"History",value:"history"}], false)
total = CalculatedMetric("trip", "Estimated total", "stay_amount * nights + extras_total", "currency", "USD", 2)`;
