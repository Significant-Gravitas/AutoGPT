import prisma.enums
import prisma.types


def installable_store_version_where() -> prisma.types.StoreListingVersionWhereInput:
    return {
        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        "isDeleted": False,
        "isAvailable": True,
        "StoreListing": {"is": {"isDeleted": False}},
    }
