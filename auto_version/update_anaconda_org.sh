#! /usr/bin/env sh
set -e

# Set up help text and options
function usage {
    echo
    echo "Usage: update_anaconda_org.sh [conda_dir [conda_conversions [update_pypi]]]"
    echo
    echo '    conda_dir: Directory in which .binstar.yml is found, and any conversions are stored.  Defaults to `.`.'
    echo '    conda_conversions: Comma-separated list of platforms to `conda convert` to.  Defaults to empty.'
    echo '        Can take values from {osx-64,linux-32,linux-64,win-32,win-64,all}.'
    echo '    update_pypi: If given as `--pypi`, run `python setup.py register sdist upload`.  Defaults to `--no-pypi`.'
    exit 1
}
if [[ $# == 1 && ( $1 == "-h" || $1 == "--help" ) ]]; then
    usage;
fi

# Parse the input options
CONDA_DIR=${1:-.}
CONVERSION=${2:-}
IFS=',' read -ra CONVERSIONS <<< "$CONVERSION"
PYPI=${3:---no-pypi}

# If we'll be doing any conversions, make a space for them
if [ ${#CONVERSIONS[@]} != 0 ]; then
    /bin/rm -rf ${TMPDIR}/conversions
    mkdir -p ${TMPDIR}/conversions
fi

# Now loop through and do the builds
/bin/rm -rf build
CONDA_PYs=( 27 35 )
for CONDA_PY in "${CONDA_PYs[@]}"
do
    echo CONDA_PY=${CONDA_PY}
    export CONDA_PY
    conda build --no-binstar-upload ${CONDA_DIR}
    conda server upload --force `conda build ${CONDA_DIR} --output`
    for conversion in "${CONVERSIONS[@]}"; do
        conda convert -f -p ${conversion} -o ${TMPDIR}/conversions `conda build ${CONDA_DIR} --output`
    done
done

# If there were any conversions, upload them all
if [ ${#CONVERSIONS[@]} != 0 ]; then
    conda server upload --force ${TMPDIR}/conversions/*/*tar.bz2
fi

# Optionally, also upload to PyPi
if [ "$PYPI" == "--pypi" ]; then
    python setup.py register sdist upload
fi
