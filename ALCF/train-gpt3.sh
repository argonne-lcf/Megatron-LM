#!/bin/bash --login

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")

SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

echo "[train-gpt3.sh]: DIR: ${DIR}"


#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
PIDS=$(ps aux | grep -E "$USER .+ python .+ pretrain_gpt.py" | grep -v grep | awk '{print $2}')
if [ -n "${PIDS}" ]; then
  echo "Already running! Exiting!"
  exit 1
fi

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ source ./launch.sh                       ┃
#┃ which then sources ./{args.sh,setup.sh}  ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
LAUNCH_FILE="${DIR}/launch.sh"
if [[ -f "${LAUNCH_FILE}" ]]; then
  echo "source-ing ${LAUNCH_FILE}"
  # shellcheck source=./launch.sh
  source "${LAUNCH_FILE}"
else
  echo "ERROR: UNABLE TO SOURCE ${LAUNCH_FILE}"
fi

export IBV_FORK_SAFE=1
echo "****************************"
echo "USING IBV_FORK_SAFE: ${IBV_FORK_SAFE}"
echo "****************************"

setup
# singleGPU "$@" 2>&1 &
# fullNode "$@" 2>&1 &
TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
export TORCH_VERSION=$TORCH_VERSION
elasticDistributed "$@" 2>&1 &
# PID=$!
# wait $PID
