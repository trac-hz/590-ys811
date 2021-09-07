ScriptLoc=${PWD} #save script directory path as shell variable
cd LectureCodes
for i in *.py; do echo $i; python $i; done #run all python scripts in directory
cd $ScriptLoc #return to script directory
for i in *.py; do echo $i; python $i; done #run all python scripts in director
