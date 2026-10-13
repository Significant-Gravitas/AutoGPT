export const launchSchedule = `root = Workspace("Launch plan", "One person; sequential work.", [form])
form = Form("launch", "Update the plan", [Schedule("work", "Work plan", [{label:"Monday",windows:[{start:"09:00",end:"12:00"},{start:"13:00",end:"17:00"}]},{label:"Tuesday",windows:[{start:"09:00",end:"12:00"},{start:"13:00",end:"17:00"}]}], [{id:"research",title:"Research",minutes:240,done:false,splittable:true},{id:"design",title:"Design",minutes:300,done:false,splittable:true},{id:"review",title:"Review",minutes:180,done:false,splittable:true},{id:"delivery",title:"Delivery",minutes:60,done:false,splittable:false}])], "Discuss this plan", "Review the edited schedule")`;

export const revisedLaunch = launchSchedule
  .replace(
    'label:"Monday",windows:[{start:"09:00",end:"12:00"},{start:"13:00",end:"17:00"}]',
    'label:"Monday",windows:[{start:"14:00",end:"17:00"}]',
  )
  .replace("minutes:240,done:false", "minutes:240,done:true")
  .replace("minutes:300", "minutes:420");

export const rehearsalSchedule = `root = Workspace("Rehearsal", "B must finish before 15:10. Setup is complete.", [form])
form = Form("rehearsal", "Edit agenda", [Schedule("agenda", "Rehearsal agenda", [{label:"Today",windows:[{start:"14:20",end:"17:00"}]}], [{id:"setup",title:"Setup",minutes:20,done:true,splittable:false},{id:"b",title:"Rehearsal B",minutes:40,done:false,splittable:false,deadline:{day:0,time:"15:10"}},{id:"a",title:"Rehearsal A",minutes:45,done:false,splittable:false},{id:"break",title:"Break",minutes:10,done:false,splittable:false},{id:"review",title:"Joint review",minutes:30,done:false,splittable:false},{id:"pack",title:"Pack-down",minutes:15,done:false,splittable:false}])], "Discuss agenda", "Review this agenda")`;
