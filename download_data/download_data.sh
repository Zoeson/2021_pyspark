#!/bin/bash

date=$1
hour=$2
tmpPostfix=$3

hdfsPath="/user/recom/recall/mind/"$date"/"$hour"/click_log"
echo "hdfsPath:"$hdfsPath
localPath="/data/recom/recall/download_data/"$date"/"$hour
localTmpPath=$localPath$tmpPostfix

hadoop fs -get $hdfsPath $localTmpPath