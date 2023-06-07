@echo off

REM Create a new branch and switch to it
git checkout -b private

REM Create a new .private directory and move .user.workspace to it
mkdir .private
git mv .user.workspace .private/

REM Add the changes to the staging area
git add .

REM Commit the changes with a commit message
git commit -m "Make .user.workspace private"

REM Push the changes to the new private branch
git push --set-upstream origin private
