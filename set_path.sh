source ./config

echo "Current PROJ_ROOT env variable set to: "$PROJ_ROOT
echo "(If this is not good, modify PROJ_ROOT variable in the ./config file"
echo "and call this command again)"
export PROJ_ROOT=$PROJ_ROOT

echo "Adding it to PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$PROJ_ROOT
