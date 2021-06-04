#!/bin/bash
$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--name "download_click_data" \
#--conf spark.pyspark.python=/data/recom/install/anaconda3/bin/python \
--conf spark.port.maxRetries=100 \
--conf spark.default.parallelism=1000 \
--conf spark.sql.shuffle.partitions=1000 \
--conf spark.dynamicAllocation.enable=false \
--conf spark.yarn.executor.memoryOverhead=1024 \
--conf spark.yarn.driver.memoryOverhead=1024 \
--conf spark.memory.storageFraction=0.3 \
--conf spark.shuffle.service.enabled=false  \
--conf spark.core.connection.ack.wait.timeout=900 \
--conf spark.shuffle.io.maxRetries=6 \
--conf spark.hadoop.validateOutputSpecs=false \
--conf spark.kryoserializer.buffer.max=1024M \
--conf spark.shuffle.io.retryWait=10 \
--conf spark.executor.extraJavaOptions="-XX:MaxPermSize=128m -XX:+PrintGC -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+PrintGCDateStamps -XX:+PrintGCApplicationStoppedTime -XX:+PrintHeapAtGC -XX:+PrintGCApplicationConcurrentTime -Xloggc:gc.log" \
--conf spark.broadcast.compress=true \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--queue "recom" \
/data/recom/recall/tfgpu/generate_click_data.py

echo "Done"


