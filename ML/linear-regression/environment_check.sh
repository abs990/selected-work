echo "--Checking Python3 version--"
pyVersion=`python3 --version`
if [[ -n "${pyVersion}" ]]; then
    echo $pyVersion
else
    echo "Python 3 not installed"
fi        
echo "--Checking pip3 version--"
pipVersion=`pip3 --version`
if [[ -n "${pipVersion}" ]]; then
    echo $pipVersion
else
    echo "Pip 3 not installed"
fi 
echo "--Checking dependencies for code--"
for libraryModule in numpy scipy pandas matplotlib
do
    echo "--Checking "$libraryModule" version--"
    libraryVersion=`pip3 list | grep $libraryModule`
    if [[ -n "${libraryVersion}" ]]; then
        echo $libraryVersion
    else
        echo $libraryModule" not installed"
    fi
done  