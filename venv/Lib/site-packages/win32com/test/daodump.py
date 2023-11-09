# import dao3032
# No longer imported here - callers responsibility to load
#
import win32com.client


def DumpDB(db, bDeep=1):
    # MUST be a DB object.
    DumpTables(db, bDeep)
    DumpRelations(db, bDeep)
    DumpAllContainers(db, bDeep)


def DumpTables(db, bDeep=1):
    for tab in db.TableDefs:
        tab = db.TableDefs(tab.Name)  # Redundant lookup for testing purposes.
        print(
            "Table %s - Fields: %d, Attributes:%d"
            % (tab.Name, len(tab.Fields), tab.Attributes)
        )
        if bDeep:
            DumpFields(tab.Fields)


def DumpFields(fields):
    for field in fields:
        print(
            "  %s, size=%d, reqd=%d, type=%d, defVal=%s"
            % (
                field.Name,
                field.Size,
                field.Required,
                field.Type,
                str(field.DefaultValue),
            )
        )


def DumpRelations(db, bDeep=1):
    for relation in db.Relations:
        print(
            "Relation %s - %s->%s"
            % (relation.Name, relation.Table, relation.ForeignTable)
        )


#### This dont work.  TLB says it is a Fields collection, but apparently not!
####            if bDeep: DumpFields(relation.Fields)


def DumpAllContainers(db, bDeep=1):
    for cont in db.Containers:
        print("Container %s - %d documents" % (cont.Name, len(cont.Documents)))
        if bDeep:
            DumpContainerDocuments(cont)


def DumpContainerDocuments(container):
    for doc in container.Documents:
        import time

        timeStr = time.ctime(int(doc.LastUpdated))
        print("  %s - updated %s (" % (doc.Name, timeStr), end=" ")
        print(doc.LastUpdated, ")")  # test the _print_ method?


def TestEngine(engine):
    import sys

    if len(sys.argv) > 1:
        dbName = sys.argv[1]
    else:
        dbName = "e:\\temp\\TestPython.mdb"
    db = engine.OpenDatabase(dbName)
    DumpDB(db)


def test():
    for progid in ("DAO.DBEngine.36", "DAO.DBEngine.35", "DAO.DBEngine.30"):
        try:
            ob = win32com.client.gencache.EnsureDispatch(progid)
        except pythoncom.com_error:
            print(progid, "does not seem to be installed")
        else:
            TestEngine(ob)
            break


if __name__ == "__main__":
    test()
