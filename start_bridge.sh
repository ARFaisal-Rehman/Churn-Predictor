#!/bin/bash

# start_bridge.sh - Script to compile and start the Java bridge server for py4j

# Create bin directory if it doesn't exist
mkdir -p bin

# Compile Java files
echo "Compiling Java files..."
javac -d bin -cp ".:lib/*" *.java

if [ $? -eq 0 ]; then
    echo "Compilation successful."
    
    # Start the Java server
    echo "Starting Java bridge server..."
    java -cp "bin:lib/*" com.riskradar.churn.JavaBridgeServer
else
    echo "Compilation failed."
    exit 1
fi