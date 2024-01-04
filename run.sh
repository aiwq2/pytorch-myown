# -----需要自行配置-----
DEVICE='0' # -1表示使用CPU，其他表示使用多少号的GPU卡
CONFIG_FILE=config/config.yaml # 配置文件
MASTER_ADDR='127.0.0.1' # Master物理机器的IP地址
MASTER_PORT=29505 # Master物理机器的开放端口号,29501
NNODES=1 # 总共几台物理机器
NODE_RANK=0 # 当前是第几台物理机器
SOCKET=eth0 # 需要绑定的网卡名称
# -----自动配置-----
TZ='Asia/Shanghai'
TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S -d "8hour") # 时间戳
# TIMESTAMP="2023_10_14_09_10_35"
DEVICE_ARR=(${DEVICE//,/ })
DEVICE_LEN=${#DEVICE_ARR[@]}
echo $TIMESTAMP
# 初始化LOCAL_RANK,RANK,WOLRD_SIZE参数
torchrun --nnodes=${NNODES} \
    --nproc_per_node=${DEVICE_LEN} \
	--master-addr=${MASTER_ADDR} \
	--master-port=${MASTER_PORT} \
	--node-rank=${NODE_RANK} \
    main.py \
    --gpu ${DEVICE} \
    --config-file ${CONFIG_FILE} \
    --datetime ${TIMESTAMP}