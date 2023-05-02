#!/bin/bash
cd "$(dirname "$0")"

sed -ie 's/mysql/sqlite/g' schema.prisma
sed -ie 's/postgres/sqlite/g' schema.prisma
sed -ie 's/@db.Text//' schema.prisma
