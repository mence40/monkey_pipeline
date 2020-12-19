#!/bin/bash

# JSON object to pass to Lambda Function
json={"\"shift\"":"\"22\"","\"msg\"":"\"ServerlessComputingWithFaaS\""}

echo "Invoking Lambda funciton using AWS CLI"
time output=`aws lambda invoke --invocation-type RequestResponse --function-name CompDP2 --region us-east-1 --payload $json /dev/stdout | head -n 1 | head -c -2 ; echo`

echo ""

echo "JSON RESULT:"
echo $output | jq
echo ""
