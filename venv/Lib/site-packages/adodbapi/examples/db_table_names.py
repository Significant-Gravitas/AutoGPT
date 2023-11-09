""" db_table_names.py -- a simple demo for ADO database table listing."""
import sys

import adodbapi

try:
    databasename = sys.argv[1]
except IndexError:
    databasename = "test.mdb"

provider = ["prv", "Microsoft.ACE.OLEDB.12.0", "Microsoft.Jet.OLEDB.4.0"]
constr = "Provider=%(prv)s;Data Source=%(db)s"

# create the connection
con = adodbapi.connect(constr, db=databasename, macro_is64bit=provider)

print("Table names in= %s" % databasename)

for table in con.get_table_names():
    print(table)
