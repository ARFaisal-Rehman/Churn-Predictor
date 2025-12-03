// JavaBridgeServer.java - Server class to start the py4j gateway

package com.riskradar.churn;

import py4j.GatewayServer;

/**
 * JavaBridgeServer starts a py4j gateway server to allow Python code
 * to interact with the Java ChurnPredictionService.
 */
public class JavaBridgeServer {
    public static void main(String[] args) {
        // Create the Java bridge object that will be accessed from Python
        JavaBridge bridge = new JavaBridge();
        
        // Create and start the gateway server
        GatewayServer server = new GatewayServer(bridge);
        server.start();
        
        System.out.println("JavaBridgeServer started. Listening for Python connections...");
        System.out.println("Server running on default port 25333");
    }
}