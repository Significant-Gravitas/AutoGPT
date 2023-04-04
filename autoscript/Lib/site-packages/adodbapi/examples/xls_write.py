import datetime

import adodbapi

try:
    import adodbapi.is64bit as is64bit

    is64 = is64bit.Python()
except ImportError:
    is64 = False  # in case the user has an old version of adodbapi
if is64:
    driver = "Microsoft.ACE.OLEDB.12.0"
else:
    driver = "Microsoft.Jet.OLEDB.4.0"
filename = "xx.xls"  # file will be created if it does not exist
extended = 'Extended Properties="Excel 8.0;Readonly=False;"'

constr = "Provider=%s;Data Source=%s;%s" % (driver, filename, extended)

conn = adodbapi.connect(constr)
with conn:  # will auto commit if no errors
    with conn.cursor() as crsr:
        try:
            crsr.execute("drop table SheetOne")
        except:
            pass  # just is case there is one already there

        # create the sheet and the header row and set the types for the columns
        crsr.execute(
            "create table SheetOne (Name varchar, Rank varchar, SrvcNum integer, Weight float,  Birth date)"
        )

        sql = "INSERT INTO SheetOne (name, rank , srvcnum, weight, birth) values (?,?,?,?,?)"

        data = ("Mike Murphy", "SSG", 123456789, 167.8, datetime.date(1922, 12, 27))
        crsr.execute(sql, data)  # write the first row of data
        crsr.execute(
            sql, ["John Jones", "Pvt", 987654321, 140.0, datetime.date(1921, 7, 4)]
        )  # another row of data
conn.close()
print("Created spreadsheet=%s worksheet=%s" % (filename, "SheetOne"))
