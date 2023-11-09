# Dump lots of info about BITS jobs.
import pythoncom
from win32com.bits import bits

states = dict(
    [
        (val, (name[13:]))
        for name, val in vars(bits).items()
        if name.startswith("BG_JOB_STATE_")
    ]
)

job_types = dict(
    [
        (val, (name[12:]))
        for name, val in vars(bits).items()
        if name.startswith("BG_JOB_TYPE_")
    ]
)

bcm = pythoncom.CoCreateInstance(
    bits.CLSID_BackgroundCopyManager,
    None,
    pythoncom.CLSCTX_LOCAL_SERVER,
    bits.IID_IBackgroundCopyManager,
)

try:
    enum = bcm.EnumJobs(bits.BG_JOB_ENUM_ALL_USERS)
except pythoncom.error:
    print("Failed to get jobs for all users - trying for current user")
    enum = bcm.EnumJobs(0)

for job in enum:
    print("Job:", job.GetDisplayName())
    print("Description:", job.GetDescription())
    print("Id:", job.GetId())
    print("State:", states.get(job.GetState()))
    print("Type:", job_types.get(job.GetType()))
    print("Owner:", job.GetOwner())
    print("Errors:", job.GetErrorCount())
    print("Created/Modified/Finished times:", [str(t) for t in job.GetTimes()])
    bytes_tot, bytes_xf, files_tot, files_xf = job.GetProgress()
    print("Bytes: %d complete of %d total" % (bytes_xf, bytes_tot))
    print("Files: %d complete of %d total" % (files_xf, files_tot))
    for f in job.EnumFiles():
        bytes, total, done = f.GetProgress()
        print("  Remote:", f.GetRemoteName())
        print("  Local:", f.GetLocalName())
        print("  Progress: %d of %d bytes - completed=%s)" % (bytes, total, done))
        print()
    print()
